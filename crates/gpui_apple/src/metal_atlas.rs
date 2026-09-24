use anyhow::{Context as _, Result};
use derive_more::{Deref, DerefMut};
use etagere::BucketedAtlasAllocator;
use gpui::{
    AtlasBackend, AtlasKey, AtlasState, AtlasTextureId, AtlasTextureKind, AtlasTextureList,
    AtlasTile, Bounds, DevicePixels, PlatformAtlas, Point, Size,
};
use metal::{CommandQueue, Device, MTLBlitOption, MTLOrigin, MTLResourceOptions, MTLSize};
use objc2::runtime::AnyObject;
use parking_lot::Mutex;
use std::{borrow::Cow, ptr::NonNull};

const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
    width: DevicePixels(1024),
    height: DevicePixels(1024),
};

// Max texture size on all modern Apple GPUs. Anything bigger than that crashes in validateWithDevice.
const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
    width: DevicePixels(16384),
    height: DevicePixels(16384),
};

/// Upper bound on pending dynamic-texture bytes held on the CPU before they are
/// flushed to the GPU. Full-texture updates coalesce, so this only accumulates
/// for partial updates while no frame is being drawn.
const MAX_PENDING_UPLOAD_BYTES: usize = 64 * 1024 * 1024;

pub struct MetalAtlas(Mutex<AtlasState<MetalAtlasTextures>>);

impl MetalAtlas {
    pub(crate) fn new(device: Device, is_apple_gpu: bool, command_queue: CommandQueue) -> Self {
        MetalAtlas(Mutex::new(AtlasState::new(MetalAtlasTextures {
            device: AssertSend(device),
            command_queue: AssertSend(command_queue),
            is_apple_gpu,
            monochrome_textures: Default::default(),
            polychrome_textures: Default::default(),
            image_textures: Default::default(),
            image_small_textures: Default::default(),
            pending_uploads: Vec::new(),
            pending_upload_bytes: 0,
        })))
    }

    /// Returns the GPU texture backing `id`, or `None` once every tile in it
    /// has been removed. A scene can still reference such a texture when a
    /// cached view replays a paint from before the image was dropped, so
    /// callers must skip those sprites rather than assume the texture exists.
    pub(crate) fn metal_texture(&self, id: AtlasTextureId) -> Option<metal::Texture> {
        Some(self.0.lock().backend.texture(id)?.metal_texture.clone())
    }

    /// Applies queued dynamic-texture uploads in `command_buffer`.
    ///
    /// Updates are recorded as GPU blits from a staging buffer instead of CPU-side
    /// `replaceRegion`, so they are ordered after previously submitted command
    /// buffers by the shared command queue and cannot race a frame that is still
    /// reading the texture.
    pub(crate) fn encode_pending_uploads(&self, command_buffer: &metal::CommandBufferRef) {
        self.0.lock().backend.encode_pending_locked(command_buffer);
    }

    /// Encodes and commits any queued uploads on their own command buffer.
    ///
    /// Uploads are consumed into a command buffer that is committed independently
    /// of the frame render pass, so a later drawing error that drops the render
    /// command buffer cannot lose already-accepted updates. Both command buffers
    /// share the renderer's queue, so the uploads still execute before the frame
    /// that samples the texture.
    pub(crate) fn commit_pending_uploads(&self) {
        if self.0.lock().backend.pending_uploads.is_empty() {
            return;
        }
        let command_queue = self.0.lock().backend.command_queue.0.clone();
        let command_buffer = command_queue.new_command_buffer().to_owned();
        self.encode_pending_uploads(&command_buffer);
        command_buffer.commit();
    }
}

/// A dynamic-texture upload queued until the next frame's command buffer.
struct PendingUpload {
    texture_id: AtlasTextureId,
    /// Device-pixel destination origin of the upload within the atlas texture.
    origin: MTLOrigin,
    size: MTLSize,
    bytes_per_row: u64,
    data: Vec<u8>,
}

struct MetalAtlasTextures {
    device: AssertSend<Device>,
    command_queue: AssertSend<CommandQueue>,
    is_apple_gpu: bool,
    monochrome_textures: AtlasTextureList<MetalAtlasTexture>,
    polychrome_textures: AtlasTextureList<MetalAtlasTexture>,
    image_textures: AtlasTextureList<MetalAtlasTexture>,
    image_small_textures: AtlasTextureList<MetalAtlasTexture>,
    pending_uploads: Vec<PendingUpload>,
    pending_upload_bytes: usize,
}

impl PlatformAtlas for MetalAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        let mut lock = self.0.lock();
        match key {
            // Dynamic textures get a dedicated texture sized exactly to the
            // update, so partial updates never spill into a neighbor tile.
            AtlasKey::DynamicTexture(_) => {
                if lock.contains(&key) {
                    return lock.get_or_insert_with(key, build);
                }
                let Some((size, _)) = build()? else {
                    return Ok(None);
                };
                drop(lock);
                // Re-lock and insert through the backend so the dedicated
                // texture is created under the same lock as the map insert.
                let mut lock = self.0.lock();
                let tile = lock
                    .backend
                    .insert_dedicated(key.texture_kind(), size)?;
                lock.insert_tile(key, tile);
                Ok(Some(tile))
            }
            _ => lock.get_or_insert_with(key, build),
        }
    }

    fn update(&self, key: &AtlasKey, bounds: Bounds<DevicePixels>, bytes: &[u8]) -> Result<()> {
        let mut lock = self.0.lock();
        let Some(tile) = lock.tile_for(key).copied() else {
            return Ok(());
        };
        let bytes_per_pixel = lock
            .backend
            .texture(tile.texture_id)
            .context("updated tile refers to a missing texture")?
            .bytes_per_pixel();
        validate_upload(tile, bounds, bytes, bytes_per_pixel)?;
        let upload_bounds = Bounds {
            origin: Point {
                x: DevicePixels(
                    tile.bounds
                        .origin
                        .x
                        .0
                        .checked_add(bounds.origin.x.0)
                        .context("texture update horizontal origin overflow")?,
                ),
                y: DevicePixels(
                    tile.bounds
                        .origin
                        .y
                        .0
                        .checked_add(bounds.origin.y.0)
                        .context("texture update vertical origin overflow")?,
                ),
            },
            size: bounds.size,
        };
        lock.backend.queue_upload(tile, upload_bounds, bytes, bytes_per_pixel);
        Ok(())
    }

    fn resource_generation(&self) -> u64 {
        0
    }

    fn max_texture_size(&self) -> Option<Size<DevicePixels>> {
        Some(MAX_ATLAS_SIZE)
    }

    fn remove(&self, key: &AtlasKey) {
        self.0.lock().remove(key);
    }
}

impl AtlasBackend for MetalAtlasTextures {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<AtlasTile> {
        let tile = self.allocate(size, kind).context("failed to allocate")?;
        let texture = self
            .texture(tile.texture_id)
            .context("allocated tile refers to a missing texture")?;
        texture.upload(tile.bounds, bytes);
        Ok(tile)
    }

    fn remove(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;
        let mut freed = false;

        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Image => &mut self.image_textures,
            AtlasTextureKind::ImageSmall => &mut self.image_small_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };

        let Some(texture_slot) = textures
            .textures
            .iter_mut()
            .find(|texture| texture.as_ref().is_some_and(|v| v.id == id))
        else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(id.index as usize);
                freed = true;
            } else {
                *texture_slot = Some(texture);
            }
        }

        if freed {
            // The texture is gone; drop its queued dynamic-texture uploads too.
            self.pending_uploads
                .retain(|upload| upload.texture_id != id);
            self.pending_upload_bytes = self
                .pending_uploads
                .iter()
                .map(|upload| upload.data.len())
                .sum();
        }
    }
}

impl MetalAtlasTextures {
    /// Inserts a dynamic texture as a dedicated texture sized exactly to it.
    fn insert_dedicated(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
    ) -> Result<AtlasTile> {
        self.allocate_dedicated(size, kind)
            .context("failed to allocate dedicated dynamic-texture")
    }

    fn allocate_dedicated(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        if size.width.0 <= 0
            || size.height.0 <= 0
            || size.width > MAX_ATLAS_SIZE.width
            || size.height > MAX_ATLAS_SIZE.height
        {
            return None;
        }

        self.push_texture_with_size(size, texture_kind)
            .allocate(size)
    }

    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        // Small images go to their own texture pages so a large render does not
        // evict a whole shared page of icons.
        const SMALL_IMAGE_TILE_MAX: i32 = 256;
        let texture_kind = if texture_kind == AtlasTextureKind::Image
            && size.width.0 <= SMALL_IMAGE_TILE_MAX
            && size.height.0 <= SMALL_IMAGE_TILE_MAX
        {
            AtlasTextureKind::ImageSmall
        } else {
            texture_kind
        };
        {
            let textures = match texture_kind {
                AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
                AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
                AtlasTextureKind::Image => &mut self.image_textures,
                AtlasTextureKind::ImageSmall => &mut self.image_small_textures,
                AtlasTextureKind::Subpixel => unreachable!(),
            };

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut MetalAtlasTexture {
        // Color textures start smaller: image tiles are sparse and a 1024px
        // first page wastes memory for icon-heavy but text-light surfaces.
        const DEFAULT_COLOR_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(512),
            height: DevicePixels(512),
        };
        let default_size = match kind {
            AtlasTextureKind::Polychrome
            | AtlasTextureKind::Image
            | AtlasTextureKind::ImageSmall => DEFAULT_COLOR_ATLAS_SIZE,
            AtlasTextureKind::Monochrome => DEFAULT_ATLAS_SIZE,
            AtlasTextureKind::Subpixel => unreachable!(),
        };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&default_size);
        self.push_texture_with_size(size, kind)
    }

    fn push_texture_with_size(
        &mut self,
        size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut MetalAtlasTexture {
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.into());
        texture_descriptor.set_height(size.height.into());
        let pixel_format;
        let usage;
        match kind {
            AtlasTextureKind::Monochrome => {
                pixel_format = metal::MTLPixelFormat::A8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Polychrome
            | AtlasTextureKind::Image
            | AtlasTextureKind::ImageSmall => {
                pixel_format = metal::MTLPixelFormat::BGRA8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Subpixel => unreachable!(),
        }
        texture_descriptor.set_pixel_format(pixel_format);
        texture_descriptor.set_usage(usage);
        // Shared memory mode can be used only on Apple GPU families
        // https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared
        texture_descriptor.set_storage_mode(if self.is_apple_gpu {
            metal::MTLStorageMode::Shared
        } else {
            metal::MTLStorageMode::Managed
        });
        let metal_texture = self.device.new_texture(&texture_descriptor);

        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Image => &mut self.image_textures,
            AtlasTextureKind::ImageSmall => &mut self.image_small_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };

        let index = texture_list.free_list.pop();

        let atlas_texture = MetalAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: etagere::BucketedAtlasAllocator::new(size_to_etagere(size)),
            metal_texture: AssertSend(metal_texture),
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix)
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut()
        }
        .unwrap()
        .as_mut()
        .unwrap()
    }

    fn texture(&self, id: AtlasTextureId) -> Option<&MetalAtlasTexture> {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
            AtlasTextureKind::Image => &self.image_textures,
            AtlasTextureKind::ImageSmall => &self.image_small_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };
        textures.textures.get(id.index as usize)?.as_ref()
    }

    /// Queues a dynamic-texture upload to be applied on the next frame's command buffer.
    fn queue_upload(
        &mut self,
        tile: AtlasTile,
        bounds: Bounds<DevicePixels>,
        bytes: &[u8],
        bytes_per_pixel: u8,
    ) {
        let is_full_update = bounds.origin == tile.bounds.origin && bounds.size == tile.bounds.size;
        if is_full_update {
            // A full-texture upload supersedes every earlier upload for this texture.
            self.pending_uploads
                .retain(|upload| upload.texture_id != tile.texture_id);
            self.pending_upload_bytes = self
                .pending_uploads
                .iter()
                .map(|upload| upload.data.len())
                .sum();
        }

        self.pending_upload_bytes = self.pending_upload_bytes.saturating_add(bytes.len());
        self.pending_uploads.push(PendingUpload {
            texture_id: tile.texture_id,
            origin: MTLOrigin {
                x: bounds.origin.x.0 as u64,
                y: bounds.origin.y.0 as u64,
                z: 0,
            },
            size: MTLSize::new(bounds.size.width.0 as u64, bounds.size.height.0 as u64, 1),
            bytes_per_row: bounds.size.width.to_bytes(bytes_per_pixel) as u64,
            data: bytes.to_vec(),
        });

        if self.pending_upload_bytes > MAX_PENDING_UPLOAD_BYTES {
            let command_queue = self.command_queue.0.clone();
            let command_buffer = command_queue.new_command_buffer().to_owned();
            self.encode_pending_locked(&command_buffer);
            command_buffer.commit();
        }
    }

    /// Encodes and clears queued uploads onto `command_buffer`.
    fn encode_pending_locked(&mut self, command_buffer: &metal::CommandBufferRef) {
        let pending_uploads = std::mem::take(&mut self.pending_uploads);
        self.pending_upload_bytes = 0;
        if pending_uploads.is_empty() {
            return;
        }

        let blit = command_buffer.new_blit_command_encoder();
        let mut staging_buffers = Vec::with_capacity(pending_uploads.len());
        for upload in pending_uploads {
            let Some(texture) = self.texture(upload.texture_id) else {
                // The texture was removed before this frame; its queued pixels are
                // no longer needed.
                continue;
            };
            let buffer = self.device.0.new_buffer_with_data(
                upload.data.as_ptr().cast(),
                upload.data.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            blit.copy_from_buffer_to_texture(
                &buffer,
                0,
                upload.bytes_per_row,
                0,
                upload.size,
                &texture.metal_texture,
                0,
                0,
                upload.origin,
                MTLBlitOption::None,
            );
            staging_buffers.push(buffer);
        }
        blit.end_encoding();

        // Keep the staging buffers alive until the GPU has consumed them.
        let staging_buffers = std::cell::Cell::new(Some(staging_buffers));
        let block = block2::RcBlock::new(move |_: NonNull<AnyObject>| {
            let _ = staging_buffers.take();
        });
        // SAFETY: Both pointee types are opaque views of the same Objective-C
        // block pointer ABI.
        unsafe {
            command_buffer.add_completed_handler(&*block2::RcBlock::as_ptr(&block).cast());
        }
    }
}

/// Validates a dynamic-texture update against its atlas tile: the origin must
/// be non-negative, dimensions positive, the region inside the tile, and the
/// byte count exactly the region size.
fn validate_upload(
    tile: AtlasTile,
    bounds: Bounds<DevicePixels>,
    bytes: &[u8],
    bytes_per_pixel: u8,
) -> Result<()> {
    anyhow::ensure!(
        bounds.origin.x.0 >= 0 && bounds.origin.y.0 >= 0,
        "texture update origin must be non-negative"
    );
    anyhow::ensure!(
        bounds.size.width.0 > 0 && bounds.size.height.0 > 0,
        "texture update size must be positive"
    );
    let right = bounds
        .origin
        .x
        .0
        .checked_add(bounds.size.width.0)
        .context("texture update horizontal bounds overflow")?;
    let bottom = bounds
        .origin
        .y
        .0
        .checked_add(bounds.size.height.0)
        .context("texture update vertical bounds overflow")?;
    anyhow::ensure!(
        right <= tile.bounds.size.width.0 && bottom <= tile.bounds.size.height.0,
        "texture update exceeds the allocated tile bounds"
    );
    let expected_len = usize::try_from(bounds.size.width.0)
        .ok()
        .and_then(|width| {
            usize::try_from(bounds.size.height.0)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel as usize))
        .context("texture update byte size overflow")?;
    anyhow::ensure!(
        bytes.len() == expected_len,
        "texture update contains {} bytes, expected {expected_len}",
        bytes.len()
    );
    Ok(())
}

struct MetalAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    metal_texture: AssertSend<metal::Texture>,
    live_atlas_keys: u32,
}

impl MetalAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: point_from_etagere(allocation.rectangle.min),
                size,
            },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(&self, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let region = metal::MTLRegion::new_2d(
            bounds.origin.x.into(),
            bounds.origin.y.into(),
            bounds.size.width.into(),
            bounds.size.height.into(),
        );
        self.metal_texture.replace_region(
            region,
            0,
            bytes.as_ptr() as *const _,
            bounds.size.width.to_bytes(self.bytes_per_pixel()) as u64,
        );
    }

    fn bytes_per_pixel(&self) -> u8 {
        use metal::MTLPixelFormat::*;
        match self.metal_texture.pixel_format() {
            A8Unorm | R8Unorm => 1,
            RGBA8Unorm | BGRA8Unorm => 4,
            _ => unimplemented!(),
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn point_from_etagere(value: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels::from(value.x),
        y: DevicePixels::from(value.y),
    }
}

#[derive(Deref, DerefMut)]
struct AssertSend<T>(T);

unsafe impl<T> Send for AssertSend<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::PlatformAtlas;
    use std::borrow::Cow;

    fn create_atlas() -> Option<MetalAtlas> {
        let device = metal::Device::system_default()?;
        let command_queue = device.new_command_queue();
        Some(MetalAtlas::new(device, true, command_queue))
    }

    fn make_image_key(image_id: usize, frame_index: usize) -> AtlasKey {
        AtlasKey::Image(gpui::RenderImageParams {
            image_id: gpui::ImageId(image_id),
            frame_index,
        })
    }

    fn insert_tile(atlas: &MetalAtlas, key: AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
    }

    #[test]
    fn test_remove_clears_stale_keys_from_tiles_by_key() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };

        let key_a = make_image_key(1, 0);
        let key_b = make_image_key(2, 0);
        let key_c = make_image_key(3, 0);

        let tile_a = insert_tile(&atlas, key_a.clone(), small);
        let tile_b = insert_tile(&atlas, key_b.clone(), small);
        let tile_c = insert_tile(&atlas, key_c.clone(), small);

        assert_eq!(tile_a.texture_id.kind, AtlasTextureKind::ImageSmall);
        assert_eq!(tile_a.texture_id, tile_b.texture_id);
        assert_eq!(tile_b.texture_id, tile_c.texture_id);

        // Remove A: texture still has B and C, so it stays.
        // The key for A must be removed from tiles_by_key.
        atlas.remove(&key_a);

        // Remove B: texture still has C.
        atlas.remove(&key_b);

        // Remove C: texture becomes unreferenced and is deleted.
        atlas.remove(&key_c);

        // Re-inserting A must allocate a fresh tile on a new texture,
        // NOT return a stale tile referencing the deleted texture.
        let tile_a2 = insert_tile(&atlas, key_a, small);

        // The texture must actually exist — this would panic before the fix.
        assert!(atlas.metal_texture(tile_a2.texture_id).is_some());
    }

    #[test]
    fn test_metal_texture_is_none_after_last_tile_removed() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let key = make_image_key(1, 0);
        let tile = insert_tile(
            &atlas,
            key.clone(),
            Size {
                width: DevicePixels(64),
                height: DevicePixels(64),
            },
        );
        assert!(atlas.metal_texture(tile.texture_id).is_some());

        // A scene built before the removal may still carry `tile`; looking its
        // texture up must report the gap instead of panicking.
        atlas.remove(&key);
        assert!(atlas.metal_texture(tile.texture_id).is_none());
    }

    #[test]
    fn test_remove_deallocates_tile_space_for_reuse() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let big_key_a = make_image_key(2, 0);
        let big_key_b = make_image_key(3, 0);

        let tile_a = insert_tile(&atlas, big_key_a.clone(), big);
        assert_eq!(tile_a.texture_id.kind, AtlasTextureKind::Image);

        atlas.remove(&big_key_a);
        let tile_b = insert_tile(&atlas, big_key_b, big);
        assert_eq!(tile_b.texture_id, tile_a.texture_id);
    }

    #[test]
    fn test_remove_nonexistent_key_is_noop() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let key = make_image_key(999, 0);
        atlas.remove(&key);
    }
}
