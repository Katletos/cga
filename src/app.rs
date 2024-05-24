use crate::camera::{Camera, CameraLocation};
use crate::scanline::{rasterize_triangle, Vertex};
use egui::{special_emojis, Pos2, Ui};

use egui::{
    epaint::ImageDelta, pos2, Color32, ColorImage, Frame, Rect, Sense, TextureId, TextureOptions,
};

use nalgebra::{Isometry3, Matrix3, Normed, Perspective3, Point2, Point3, RealField, Vector2, Vector3, Vector4};
use std::fs::{self};
use std::ops::Index;
use std::time::{Duration, Instant};
use std::{thread, vec};
use wavefront_obj::obj::ObjSet;

pub struct RenderBuffer {
    pub image: ColorImage,
    pub depth_buffer: Vec<f32>,
}

#[derive(Default)]
pub struct InputState {
    drag_start: Option<Pos2>,
    drag_delta: Option<(f32, f32)>,
    forward: bool,
    backward: bool,
}

pub struct RenderOptions {
    render_verticies: bool,
    render_normal_map: bool,
    render_depth_buffer: bool,
    mip_mapping: bool,
    ambient: f32,
    diffuse: f32,
    trilinear_filtration: bool,
    anisatropic: bool,
    mip_mapping_level_diff: f32,
}

pub struct MipPyramid {
    levels: Vec<ColorImage>,
}

impl MipPyramid {
    pub fn new(image: ColorImage) -> Self {
        let [start_width, start_height] = image.size;
        assert!(start_width == start_height);
        assert!(start_width.is_power_of_two());
        let mut images = vec![image];
        let mut size = start_width;

        let mip_colors = [
            Color32::RED,
            Color32::GREEN,
            Color32::BLUE,
            Color32::GRAY,
            Color32::GOLD,
            Color32::WHITE,
            Color32::DARK_GREEN,
            Color32::LIGHT_GREEN,
            Color32::DARK_BLUE,
            Color32::RED,
            Color32::GREEN,
            Color32::BLUE,
        ];

        while size > 1 {
            size /= 2;
            let mut new_image = ColorImage::new([size, size], Color32::BLACK);
            for x in 0..size {
                for y in 0..size {
                    let old_x = x * 2;
                    let old_y = y * 2;
                    let p1 = Vector4::from(
                        images.last().unwrap()[(old_x, old_y)].to_normalized_gamma_f32(),
                    )
                    .xyz();
                    let p2 = Vector4::from(
                        images.last().unwrap()[(old_x + 1, old_y)].to_normalized_gamma_f32(),
                    )
                    .xyz();
                    let p3 = Vector4::from(
                        images.last().unwrap()[(old_x + 1, old_y + 1)].to_normalized_gamma_f32(),
                    )
                    .xyz();
                    let p4 = Vector4::from(
                        images.last().unwrap()[(old_x, old_y + 1)].to_normalized_gamma_f32(),
                    )
                    .xyz();
                    let new_color = (p1 + p2 + p3 + p4) / 4.0;

                    let new_color = Color32::from_rgb(
                        (new_color.x * 255.0) as u8,
                        (new_color.y * 255.0) as u8,
                        (new_color.z * 255.0) as u8,
                    );

                    //let new_color = mip_colors[images.len() - 1];
                    new_image[(x, y)] = new_color;
                }
            }
            images.push(new_image);
        }
        MipPyramid { levels: images }
    }

    pub fn sample_anisatropic(
        &self,
        u: f32,
        v: f32,
        xu: f32,
        xv: f32,
        yu: f32,
        yv: f32,
    ) -> Color32 {
        let uv_x = Vector2::new(xu, xv) * 4095.0 * 4.0;
        let uv_y = Vector2::new(yu, yv) * 4095.0 * 4.0;

        let overall_delta = f32::min(uv_x.norm(), uv_y.norm());
        let level = overall_delta.log2() + 1.0;

        let du: f32;
        let dv: f32;

        if uv_x.norm_squared() > uv_y.norm_squared() {
            du = xu / 2.0;
            dv = xv / 2.0;
        } else {
            du = yu / 2.0;
            dv = yv / 2.0;
        }

        let mut samples = Vector3::<f32>::default();
        for i in 0..=2 {
            if i == 1 {
                continue;
            }
            let sample = self.index_trilinear(u + du*(i as f32 - 1.0), v + dv*(i as f32 - 1.0), level);
            let p1 = Vector4::from(sample.to_normalized_gamma_f32()).xyz();
            samples += p1;
        }
        //top_sample = self.index_trilinear(u + yu, v + yv, level);
        //bottom_sample = self.index_trilinear(u - yu, v - yv, level);

        //let p1 = Vector4::from(top_sample.to_normalized_gamma_f32()).xyz();
        //let p2 = Vector4::from(bottom_sample.to_normalized_gamma_f32()).xyz();
        
        //assert!(p1 != p2, "{uv_x:?} {uv_y:?} {level}");
        let new_color = samples / 2.0;//(p1 + p2) / 2.0;

        let new_color = Color32::from_rgb(
            (new_color.x * 255.0) as u8,
            (new_color.y * 255.0) as u8,
            (new_color.z * 255.0) as u8,
        );
        return new_color;
    }

    pub fn index_level(&self, u: f32, v: f32, level: usize) -> Color32 {
        let level = level.min(self.levels.len() - 1);
        let access_u = (u / 2u32.pow(level as u32) as f32) as usize;
        let access_v = (v / 2u32.pow(level as u32) as f32) as usize;
        self.levels[level][(access_u, access_v)]
    }

    pub fn index_bilinear(&self, u: f32, v: f32, level: usize) -> Color32 {
        let level = level.min(self.levels.len() - 1);
        let access_u = u / 2u32.pow(level as u32) as f32;
        let access_v = v / 2u32.pow(level as u32) as f32;
        let access_u = access_u % (2.0f32.powi((12 - level) as i32) - 1.0);
        let access_v = access_v % (2.0f32.powi((12 - level) as i32) - 1.0);

        //dbg!(level, access_u, access_v, u, v);

        let p1 = self.levels[level][(access_u.floor() as usize, access_v.floor() as usize)];
        let p2 = self.levels[level][(access_u.floor() as usize, access_v.floor() as usize)];
        let p3 = self.levels[level][(access_u.floor() as usize, access_v.floor() as usize)];
        let p4 = self.levels[level][(access_u.floor() as usize, access_v.floor() as usize)];

        bilinear_filter(p1, p2, p3, p4, Point2::new(access_u, access_v))
    }

    pub fn index_trilinear(&self, u: f32, v: f32, level: f32) -> Color32 {
        let level = level.clamp(0.0, (self.levels.len() - 1) as f32);

        let lower = self.index_bilinear(u, v, level.floor() as usize);
        let upper = self.index_bilinear(u, v, level.ceil() as usize);

        let l = Vector4::from(lower.to_normalized_gamma_f32()).xyz();
        let u = Vector4::from(upper.to_normalized_gamma_f32()).xyz();

        let t = level % 1.0;
        let f = (1.0 - t) * l + t * u;

        Color32::from_rgb(
            (f.x * 255.0) as u8,
            (f.y * 255.0) as u8,
            (f.z * 255.0) as u8,
        )
    }
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            render_verticies: false,
            render_normal_map: false,
            render_depth_buffer: false,
            mip_mapping: false,
            mip_mapping_level_diff: 0.0f32,
            trilinear_filtration: false,
            ambient: 0.2,
            diffuse: 0.8,
            anisatropic: false,
        }
    }
}

impl RenderBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        RenderBuffer {
            image: ColorImage::new([width, height], Color32::BLACK),
            depth_buffer: vec![f32::INFINITY; width * height],
        }
    }

    pub fn resize_buffer(&mut self, width: usize, height: usize) {
        if self.image.width() != width || self.image.height() != height {
            self.image = ColorImage::new([width, height], Color32::BLACK);
            self.depth_buffer.resize(width * height, f32::INFINITY);
        }
    }

    pub fn reset_buffer(&mut self) {
        self.image.pixels.fill(Color32::BLACK);
        self.depth_buffer.fill(f32::INFINITY);
    }
}

pub struct DickDrawingApp {
    render_texture_canvas: TextureId,
    buffer: RenderBuffer,
    render_options: RenderOptions,

    global_light: Vector3<f32>,
    diffuse_texture: MipPyramid,
    normal_texture: ColorImage,
    input: InputState,
    model: ObjSet,
    camera_location: CameraLocation,
    specular_texture: ColorImage,
}

impl DickDrawingApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_texture_canvas = cc.egui_ctx.tex_manager().write().alloc(
            "app".parse().unwrap(),
            ColorImage::default().into(),
            TextureOptions::NEAREST,
        );

        let model = wavefront_obj::obj::parse(
            //fs::read_to_string("teapot.obj").expect("Failed reading .obj file."),
            //fs::read_to_string("aboba/shovel/model.obj").expect("Failed reading .obj file."),
            //fs::read_to_string("aboba/ship/model.obj").expect("Failed reading .obj file."),
            fs::read_to_string("plane.obj3").expect("Failed reading .obj file."),
            //fs::read_to_string("plane3.obj").expect("Failed reading .obj file."),
        )
        .expect("Error in parsing .obj format.");

        //let image_data = std::fs::read("aboba/shovel/diffuse.png").unwrap();
        //let image_data = std::fs::read("aboba/deck.png").unwrap();
        //let image_data = std::fs::read("test_texture.png").unwrap();
        //let image_data = std::fs::read("checkerboard.png").unwrap();
        let image_data = std::fs::read("Cobblestone.png").unwrap();

        let image = image::load_from_memory(&image_data).unwrap();
        let image_buffer = image.to_rgba8();
        let image_buffer = image_buffer.as_raw();
        let texture_size = [image.width() as usize, image.height() as usize];
        let texture_color_image = ColorImage::from_rgba_unmultiplied(texture_size, image_buffer);

        let normal_image_data = std::fs::read("aboba/shovel/normal.png").unwrap();
        let normal_image = image::load_from_memory(&normal_image_data).unwrap();
        let normal_image_buffer = normal_image.to_rgba8();
        let normal_image_buffer = normal_image_buffer.as_raw();
        let normal_texture_size = [
            normal_image.width() as usize,
            normal_image.height() as usize,
        ];
        let normal_texture_color_image =
            ColorImage::from_rgba_unmultiplied(normal_texture_size, normal_image_buffer);

        let specular_image_data = std::fs::read("aboba/shovel/specular.png").unwrap();
        let specular_image = image::load_from_memory(&specular_image_data).unwrap();
        let specular_image_buffer = specular_image.to_rgba8();
        let specular_image_buffer = specular_image_buffer.as_raw();
        let specular_texture_size = [
            specular_image.width() as usize,
            specular_image.height() as usize,
        ];
        let specular_texture_color_image =
            ColorImage::from_rgba_unmultiplied(specular_texture_size, specular_image_buffer);

        DickDrawingApp {
            render_texture_canvas,
            //buffer: RenderBuffer::new(1024, 576),
            buffer: RenderBuffer::new(640, 480),

            input: InputState::default(),
            model,

            diffuse_texture: MipPyramid::new(texture_color_image),
            normal_texture: normal_texture_color_image,
            specular_texture: specular_texture_color_image,

            global_light: Vector3::new(1.0, 1.0, 1.0).normalize(),
            render_options: RenderOptions::default(),
            camera_location: CameraLocation::new(
                3.1415926535897932,
                0.0,
                Point3::new(7.0,  0.0, 4.0),
            ),
        }
    }
}

fn bilinear_filter(
    p1: Color32,
    p2: Color32,
    p3: Color32,
    p4: Color32,
    point: Point2<f32>,
) -> Color32 {
    let p1 = Vector4::from(p1.to_normalized_gamma_f32()).xyz();
    let p2 = Vector4::from(p2.to_normalized_gamma_f32()).xyz();
    let p3 = Vector4::from(p3.to_normalized_gamma_f32()).xyz();
    let p4 = Vector4::from(p4.to_normalized_gamma_f32()).xyz();

    let x = point.x % 1.0;
    let y = point.y % 1.0;

    let top = (1.0 - x) * p1 + x * p2;
    let bottom = (1.0 - x) * p3 + x * p4;
    let end = (1.0 - y) * top + y * bottom;

    Color32::from_rgb(
        (end.x * 255.0) as u8,
        (end.y * 255.0) as u8,
        (end.z * 255.0) as u8,
    )
}

impl eframe::App for DickDrawingApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default()
            .frame(Frame::none())
            .show(ctx, |ui| {
                let frame_time = Duration::from_secs_f32(1.0 / 60.0);
                let start = Instant::now();
                self.controls(ctx, ui);
                self.render_image(ctx, ui);
                self.debug_window(ctx, ui);
                let duration = start.elapsed();
                if let Some(to_wait) = frame_time.checked_sub(duration) {
                    thread::sleep(to_wait);
                }
                ctx.request_repaint();
            });
    }
}

impl DickDrawingApp {
    fn controls(&mut self, _ctx: &egui::Context, ui: &mut Ui) {
        ui.input(|input| {
            /*
            for event in &input.raw.events {
                if let Event::Key { key, physical_key, pressed, repeat, modifiers } = event {
                    if !repeat {
                        match key.
                    }
                }
            } */
            let forward_pressed = input.key_pressed(egui::Key::W);
            let forward_released = input.key_released(egui::Key::W);

            let backward_pressed = input.key_pressed(egui::Key::S);
            let backward_released = input.key_released(egui::Key::S);

            if forward_pressed {
                self.input.forward = true;
            } else if forward_released {
                self.input.forward = false;
            }

            if backward_pressed {
                self.input.backward = true;
            } else if backward_released {
                self.input.backward = false;
            }

            if let (Some(drag_start), Some(interact_pos)) =
                (self.input.drag_start, input.pointer.interact_pos())
            {
                let delta = drag_start - interact_pos;
                self.input.drag_delta = Some((delta.x, delta.y));
            } else {
                self.input.drag_delta = None;
            }

            if input.pointer.primary_down() {
                self.input.drag_start = input.pointer.interact_pos();
            } else {
                self.input.drag_start = None;
            }
        });

        let speed = 0.1;
        let mouse_speed = 0.01;
        if self.input.forward {
            self.camera_location.world_pos += self.camera_location.get_look_direction() * speed;
        }

        if self.input.backward {
            self.camera_location.world_pos -= self.camera_location.get_look_direction() * speed;
        }

        if let Some((dx, dy)) = self.input.drag_delta {
            self.camera_location.yaw += dx * mouse_speed;
            self.camera_location.pitch += -dy * mouse_speed;
        }
    }

    fn debug_window(&mut self, ctx: &egui::Context, ui: &mut Ui) {
        egui::Window::new("Debug data").show(ctx, |ui| {
            ui.label(format!("Pitch: {}", self.camera_location.pitch));
            ui.label(format!("Yaw: {}", self.camera_location.yaw));
            ui.label(format!("Eye: {}", self.camera_location.world_pos));
            ui.checkbox(&mut self.render_options.render_verticies, "Render vertices");
            ui.checkbox(&mut self.render_options.render_normal_map, "Normal map");
            ui.checkbox(&mut self.render_options.render_depth_buffer, "Depth buffer");
            ui.checkbox(&mut self.render_options.mip_mapping, "Mip mapping");
            ui.add(
                egui::Slider::new(&mut self.render_options.mip_mapping_level_diff, -4.0..=4.0)
                    .text("Mip mapping level diff"),
            );
            ui.checkbox(
                &mut self.render_options.trilinear_filtration,
                "Trilinear filtration",
            );
            ui.checkbox(&mut self.render_options.anisatropic, "Anisatropic");
            ui.add(egui::Slider::new(&mut self.global_light.x, -1.0..=1.0).text("X"));
            ui.add(egui::Slider::new(&mut self.global_light.y, -1.0..=1.0).text("Y"));
            ui.add(egui::Slider::new(&mut self.global_light.z, -1.0..=1.0).text("Z"));

            ui.add(
                egui::Slider::new(&mut self.render_options.ambient, 0.0..=1.0).text("Ambient: "),
            );
            ui.add(
                egui::Slider::new(&mut self.render_options.diffuse, 0.0..=1.0).text("Diffuse: "),
            );
            if ui.button("teleport").clicked() {
                self.camera_location.world_pos.y = 100.0;
            };
            self.global_light = self.global_light.normalize();
        });
    }

    fn render_image(&mut self, ctx: &egui::Context, ui: &mut Ui) {
        let (response, painter) = ui.allocate_painter(ui.ctx().screen_rect().size(), Sense::drag());

        let w = painter.clip_rect().width() as usize;
        let h = painter.clip_rect().height() as usize;
        //let w = self.buffer.image.size[0];
        //let h = self.buffer.image.size[0];

        self.buffer.resize_buffer(w, h);
        painter.image(
            self.render_texture_canvas,
            painter.clip_rect(),
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        self.buffer.reset_buffer();

        let projection = Perspective3::new(w as f32 / h as f32, 90.0f32.to_radians(), 0.1, 1000.0);

        let camera = Camera::new(&self.camera_location);

        let model = Isometry3::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::y() * std::f32::consts::FRAC_2_PI,
        );

        let model_view = camera.look_at * model;

        let to_screen_space = |vertex: Point3<f32>| {
            let m4 = projection.as_matrix() * model_view.to_homogeneous();
            let v = Vector4::new(vertex.x, vertex.y, vertex.z, 1.0);
            let a = m4 * v;
            let saved_v = a.w;
            let a = a / saved_v;
            let mut a = Point3::from(a.xyz());
            a.z = (1.0 + a.z) / 2.0;
            (a, saved_v)
        };

        let to_world_space = |normal: Vector3<f32>| model_view.transform_vector(&normal);
        let to_world_space_p = |p: Point3<f32>| model_view.transform_point(&p);

        let to_p3 = |v: wavefront_obj::obj::Vertex| Point3::new(v.x as f32, v.y as f32, v.z as f32);
        let to_v3 =
            |v: wavefront_obj::obj::Vertex| Vector3::new(v.x as f32, v.y as f32, v.z as f32);

        let to_uv3 =
            |v: wavefront_obj::obj::TVertex| Vector3::new(v.u as f32, v.v as f32, v.w as f32);

        let to_viewport = |p: Point3<f32>| {
            Point3::new(
                (p.x + 1.0) * 0.5 * w as f32,
                (1.0 - p.y) * 0.5 * h as f32,
                p.z,
            )
        };

        let in_bounds = |point: &Point3<f32>| {
            (-1.0..1.0).contains(&point.x)
                && (-1.0..1.0).contains(&point.y)
                && (0.0..1.0).contains(&point.z)
        };

        let is_visible_triangle = |p1: &Point3<f32>, p2: &Point3<f32>, p3: &Point3<f32>| {
            in_bounds(p1) || in_bounds(p2) || in_bounds(p3)
        };

        println!("haha funny rasteriizing");
        for object in &self.model.objects {
            for geometry in &object.geometry {
                let mut tangents = vec![Vector3::default(); object.vertices.len()];

                for shape in &geometry.shapes {
                    if let wavefront_obj::obj::Primitive::Triangle(ia, ib, ic) = shape.primitive {
                        let pos0 = to_p3(object.vertices[ia.0]);
                        let pos1 = to_p3(object.vertices[ib.0]);
                        let pos2 = to_p3(object.vertices[ic.0]);

                        let uv0 = to_uv3(object.tex_vertices[ia.1.unwrap()]);
                        let uv1 = to_uv3(object.tex_vertices[ib.1.unwrap()]);
                        let uv2 = to_uv3(object.tex_vertices[ic.1.unwrap()]);

                        let delta_pos1 = pos1 - pos0;
                        let delta_pos2 = pos2 - pos0;

                        let delta_uv1 = uv1 - uv0;
                        let delta_uv2 = uv2 - uv0;

                        let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                        let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;

                        tangents[ia.0] += tangent;
                        tangents[ib.0] += tangent;
                        tangents[ic.0] += tangent;
                        tangents[ia.0] = tangents[ia.0].normalize();
                        tangents[ib.0] = tangents[ib.0].normalize();
                        tangents[ic.0] = tangents[ic.0].normalize();
                    }
                }

                for shape in &geometry.shapes {
                    if let wavefront_obj::obj::Primitive::Triangle(ia, ib, ic) = shape.primitive {
                        let av = to_p3(object.vertices[ia.0]);
                        let bv = to_p3(object.vertices[ib.0]);
                        let cv = to_p3(object.vertices[ic.0]);

                        let (a, aw) = to_screen_space(av);
                        let (b, bw) = to_screen_space(bv);
                        let (c, cw) = to_screen_space(cv);

                        if ((b - a).cross(&(c - a)).z).is_sign_negative() {
                            continue;
                        }

                        let an = to_world_space(to_v3(object.normals[ia.2.unwrap()]));
                        let bn = to_world_space(to_v3(object.normals[ib.2.unwrap()]));
                        let cn = to_world_space(to_v3(object.normals[ic.2.unwrap()]));

                        let a_world = to_world_space_p(av);
                        let b_world = to_world_space_p(bv);
                        let c_world = to_world_space_p(cv);

                        let mut uv_a = to_uv3(object.tex_vertices[ia.1.unwrap()]);
                        let mut uv_b = to_uv3(object.tex_vertices[ib.1.unwrap()]);
                        let mut uv_c = to_uv3(object.tex_vertices[ic.1.unwrap()]);

                        let ta = tangents[ia.0];
                        let tb = tangents[ib.0];
                        let tc = tangents[ic.0];

                        uv_a.z = aw;
                        uv_b.z = bw;
                        uv_c.z = cw;

                        if self.render_options.render_verticies {
                            for point_coords in [a, b, c] {
                                if (-1.0..1.0).contains(&point_coords.z)
                                    && (-1.0..1.0).contains(&point_coords.x)
                                    && (-1.0..1.0).contains(&point_coords.y)
                                {
                                    let z = (point_coords.z + 1.0) / 2.0;
                                    let radius = 10.0 / z;
                                    dbg!(radius);
                                    dbg!(point_coords);
                                    painter.circle(
                                        pos2(
                                            (point_coords.x + 1.0) / 2.0 * w as f32,
                                            (1.0 - point_coords.y) / 2.0 * h as f32,
                                        ),
                                        radius,
                                        Color32::WHITE,
                                        (1.0, Color32::YELLOW),
                                    );
                                }
                            }
                        }

                        let vp_a = to_viewport(a);
                        let vp_b = to_viewport(b);
                        let vp_c = to_viewport(c);

                        if !is_visible_triangle(&a, &b, &c) {
                            continue;
                        }

                        let vertex_a = Vertex::new(vp_a, an, uv_a, ta, Color32::YELLOW, a_world);
                        let vertex_b = Vertex::new(vp_b, bn, uv_b, tb, Color32::YELLOW, b_world);
                        let vertex_c = Vertex::new(vp_c, cn, uv_c, tc, Color32::YELLOW, c_world);

                        rasterize_triangle(
                            vertex_a,
                            vertex_b,
                            vertex_c,
                            &mut |point,
                                  rgb,
                                  uv,
                                  tangent,
                                  world_pos,
                                  delta_uvx,
                                  delta_uvy,
                                  depth| {
                                let x = point.x;
                                let y = point.y;

                                if (0..w as isize).contains(&x) && (0..h as isize).contains(&y) {
                                    let world_pos: Point3<f32> =
                                        Point3::new(world_pos.0, world_pos.1, world_pos.2);

                                    let normal = Vector3::new(rgb.0, rgb.1, rgb.2);
                                    let u = uv.0 / uv.2;
                                    let v = uv.1 / uv.2;
                                    let v = 1.0 - v;

                                    let u_ix = delta_uvx.0 / delta_uvx.2;
                                    let v_ix = delta_uvx.1 / delta_uvx.2;

                                    let u_iy = delta_uvy.0 / delta_uvy.2;
                                    let v_iy = delta_uvy.1 / delta_uvy.2;

                                    let tu_f = (u * 4094.0) % 4093.0;
                                    let tv_f = (v * 4094.0) % 4093.0;

                                    let (tu, tv) = (
                                        (tu_f as usize).clamp(0, 4095),
                                        (tv_f as usize).clamp(0, 4095),
                                    );

                                    let orig_v = 1.0 - v;

                                    let step_size = 0.5;
                                    let x_u = u - u_ix;
                                    let x_v = orig_v - v_ix;
                                    let y_u = u - u_iy;
                                    let y_v = orig_v - v_iy;

                                    let delta_x = f32::max((x_u).abs(), (x_v).abs()) * 4095.0;
                                    let delta_y = f32::max((y_u).abs(), (y_v).abs()) * 4095.0;

                                    let overall_delta = f32::max(delta_x, delta_y) / step_size;

                                    let mip_level = if self.render_options.mip_mapping {
                                        overall_delta.log2() + self.render_options.mip_mapping_level_diff
                                    } else {
                                        0.0
                                    };

                                    let diffuse_color = if self.render_options.anisatropic {
                                        self.diffuse_texture
                                            .sample_anisatropic(tu_f, tv_f, x_u, x_v, y_u, y_v)
                                    } else if self.render_options.trilinear_filtration {
                                        self.diffuse_texture.index_trilinear(tu_f, tv_f, mip_level)
                                    } else {
                                        self.diffuse_texture.index_bilinear(
                                            tu_f,
                                            tv_f,
                                            mip_level as usize,
                                        )
                                    };

                                    let normal_texture = *self.normal_texture.index((tu, tv));
                                    let specular_texture = *self.specular_texture.index((tu, tv));
                                    let specular_texture = specular_texture.r() as f32 / 255.0;
                                    let specular_texture = 1.0;

                                    let normal_vector = if self.render_options.render_normal_map {
                                        let tangent = Vector3::new(tangent.0, tangent.1, tangent.2);
                                        let bitan = normal.cross(&tangent);
                                        let tbn = Matrix3::from_columns(&[tangent, bitan, normal]);

                                        let normal_from_texture = Vector3::new(
                                            (normal_texture.r() as f32 / 255.0) * 2.0 - 1.0,
                                            (normal_texture.g() as f32 / 255.0) * 2.0 - 1.0,
                                            (normal_texture.b() as f32 / 255.0) * 2.0 - 1.0,
                                        );

                                        tbn * normal_from_texture
                                    } else {
                                        normal
                                    };

                                    let view_direction =
                                        (world_pos - self.camera_location.world_pos).normalize();
                                    let dot = normal_vector.dot(&self.global_light);
                                    let caster = 1.0;
                                    let color_f32 = diffuse_color.to_normalized_gamma_f32();
                                    let color_f32 =
                                        Vector3::new(color_f32[0], color_f32[1], color_f32[2]);

                                    let ambient = color_f32 * self.render_options.ambient;
                                    let diffuse = color_f32
                                        * self.render_options.diffuse
                                        * dot.max(0.0)
                                        * caster;
                                    let tmp_vec = -2.0 * dot * normal_vector;
                                    let reflect_light = (self.global_light + tmp_vec).normalize();
                                    let spec =
                                        reflect_light.dot(&view_direction).max(0.0).powf(42.0);
                                    let material_specular = specular_texture;
                                    let specular = material_specular * color_f32 * spec;

                                    let triangle_color = ambient + diffuse + specular;

                                    let triangle_color = Color32::from_rgb(
                                        (triangle_color.x * 255.0) as u8,
                                        (triangle_color.y * 255.0) as u8,
                                        (triangle_color.z * 255.0) as u8,
                                    );

                                    /* let triangle_color = Color32::from_rgb(
                                        (delta_y * 255.0) as u8,
                                        (delta_y * 255.0) as u8,
                                        0,
                                        //(delta_x * 255.0) as u8,
                                        //(delta_uvx.1 / delta_uvx.2 * 128.0) as u8,
                                    ); */

                                    if depth > 0.0
                                        && self.buffer.depth_buffer[w * y as usize + x as usize]
                                            > depth
                                    {
                                        unsafe {
                                            // image.pixels.get_unchecked_mut(w * y as usize + x as usize) = triangle_color;
                                            // depth_buffer.get_unchecked_mut(w * y as usize + x as usize) = depth;
                                            if (0..w as isize).contains(&x)
                                                && (0..h as isize).contains(&y)
                                            {
                                                self.buffer.image.pixels
                                                    [w * y as usize + x as usize] = triangle_color;
                                                self.buffer.depth_buffer
                                                    [w * y as usize + x as usize] = depth;
                                            }
                                        }
                                    };
                                }
                            },
                        );
                    }
                }
            }
        }
        println!("finished raster :-)))))))))");

        if self.render_options.render_depth_buffer {
            for (image_pixel, depth_buffer_pixel) in self
                .buffer
                .image
                .pixels
                .iter_mut()
                .zip(self.buffer.depth_buffer.iter())
            {
                *image_pixel = Color32::from_gray((depth_buffer_pixel * 255.0) as u8);
            }
        }

        //let image_buffer
        ui.ctx().tex_manager().write().set(
            self.render_texture_canvas,
            ImageDelta::full(self.buffer.image.clone(), TextureOptions::NEAREST),
        );
    }
}
