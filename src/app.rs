use crate::bresenham::{Bresenham, Point};
use crate::camera::{Camera, CameraLocation};
use crate::obj_file::ObjModel;
use crate::scanline::{rasterize_triangle, Vertex};
use crate::triangle::ObjTriangle;
use eframe::App;
use egui::{Event, Pos2, Ui};
use image::GenericImageView;
use itertools::Itertools;
use rayon::prelude::*;

use egui::{
    epaint::ImageDelta, include_image, pos2, vec2, Color32, ColorImage, Direction, Frame, Image,
    Margin, Rect, Sense, TextureId, TextureOptions, Vec2,
};

use nalgebra::{Isometry3, Matrix3, Perspective3, Point2, Point3, Vector3, Vector4};
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::ops::Index;
use std::time::{Duration, Instant};
use std::{thread, vec};
use wavefront::Obj;
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
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            render_verticies: false,
            render_normal_map: true,
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
    diffuse_texture: ColorImage,
    normal_texture: ColorImage,
    input: InputState,
    model: ObjSet,
    camera_location: CameraLocation,
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
            fs::read_to_string("aboba/shovel/model.obj").expect("Failed reading .obj file."),
            //fs::read_to_string("plane.obj").expect("Failed reading .obj file."),
        )
        .expect("Error in parsing .obj format.");

        let image_data = std::fs::read("aboba/shovel/diffuse.png").unwrap();
        //let image_data = std::fs::read("aboba/deck.png").unwrap();
        let image = image::load_from_memory(&image_data).unwrap();
        let image_buffer = image.to_rgba8();
        let image_buffer = image_buffer.as_raw();
        let texture_size = [image.width() as usize, image.height() as usize];
        let texture_color_image = ColorImage::from_rgba_unmultiplied(texture_size, image_buffer);

        let normal_image_data = std::fs::read("aboba/shovel/normal.png").unwrap();
        let normal_image = image::load_from_memory(&normal_image_data).unwrap();
        let normal_image_buffer = normal_image.to_rgba8();
        let normal_image_buffer = normal_image_buffer.as_raw();
        let normal_texture_size = [normal_image.width() as usize, normal_image.height() as usize];
        let normal_texture_color_image =
            ColorImage::from_rgba_unmultiplied(normal_texture_size, normal_image_buffer);

        DickDrawingApp {
            render_texture_canvas,
            buffer: RenderBuffer::new(1024, 576),

            input: InputState::default(),
            model,

            diffuse_texture: texture_color_image,
            normal_texture: normal_texture_color_image,

            global_light: Vector3::new(1.0, 1.0, 1.0).normalize(),
            render_options: RenderOptions::default(),
            camera_location: CameraLocation::new(
                3.1415926535897932,
                0.0,
                Point3::new(0.0, 0.0, -10.0),
            ),
        }
    }
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
        });
    }

    fn render_image(&mut self, ctx: &egui::Context, ui: &mut Ui) {
        let (response, painter) = ui.allocate_painter(ui.ctx().screen_rect().size(), Sense::drag());

        let w = painter.clip_rect().width() as usize;
        let h = painter.clip_rect().height() as usize;

        self.buffer.resize_buffer(w, h);
        painter.image(
            self.render_texture_canvas,
            painter.clip_rect(),
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        self.buffer.reset_buffer();

        let far = 1000.0;
        let near = far / 100.0;
        let projection = Perspective3::new(w as f32 / h as f32, 60.0f32.to_radians(), 1.0, 10000.0);

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
            (Point3::from(a.xyz()), saved_v)
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
                && (-1.0..1.0).contains(&point.z)
        };

        let is_visible_triangle = |p1: &Point3<f32>, p2: &Point3<f32>, p3: &Point3<f32>| {
            in_bounds(p1) || in_bounds(p2) || in_bounds(p3)
        };

        println!("haha funny rasteriizing");
        for object in &self.model.objects {
            for geometry in &object.geometry {
                let mut tangents: HashMap<(usize, Option<usize>, Option<usize>), Vector3<f32>> =
                    HashMap::default();

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

                        // let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * -r;
                        tangents.insert(
                            ia,
                            (tangent + tangents.get(&ia).unwrap_or(&Vector3::default()))
                                .normalize(),
                        );
                        tangents.insert(
                            ib,
                            (tangent + tangents.get(&ib).unwrap_or(&Vector3::default()))
                                .normalize(),
                        );
                        tangents.insert(
                            ic,
                            (tangent + tangents.get(&ic).unwrap_or(&Vector3::default()))
                                .normalize(),
                        );
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

                        let ta = tangents[&ia];
                        let tb = tangents[&ib];
                        let tc = tangents[&ic];

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

                        let calc_light = |normal: Vector3<f32>, fragment: Vector3<f32>| {
                            let light_direction = self.global_light;
                            //let view_direction = self.camera_location.world_pos - fra
                        };

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
                            &mut |point, rgb, uv, tangent, depth| {
                                let x = point.x;
                                let y = point.y;
                                if (0..w).contains(&x) && (0..h).contains(&y) {
                                    let color = Color32::from_rgb(
                                        (rgb.0 * 255.0) as u8,
                                        (rgb.1 * 255.0) as u8,
                                        (rgb.2 * 255.0) as u8,
                                    );

                                    let mut normal = Vector3::new(rgb.0, rgb.1, rgb.2);

                                    let u = uv.0 / uv.2;
                                    let v = uv.1 / uv.2;

                                    let (tu, tv) = (
                                        ((u * 4094.0) as usize).clamp(0, 4095),
                                        (((1.0 - v) * 4094.0) as usize).clamp(0, 4095),
                                    );

                                    let diffuse_color = *self.diffuse_texture.index((tu, tv));
                                    let normal_texture = *self.normal_texture.index((tu, tv));

                                    let normal_vector = if self.render_options.render_normal_map {
                                        let tangent = Vector3::new(tangent.0, tangent.1, tangent.2);
                                        let bitan = normal.cross(&tangent);
                                        let tbn = Matrix3::from_columns(&[tangent, bitan, normal]);

                                        let normal_from_texture = Vector3::new(
                                            (normal_texture.r() as f32 / 255.0) * 2.0 - 1.0,
                                            (normal_texture.g() as f32 / 255.0) * 2.0 - 1.0,
                                            (normal_texture.b() as f32 / 255.0) * 2.0 - 1.0,
                                        );

                                        //let normal_from_texture = Vector3::new(0.0, 0.0, 1.0);
                                        tbn * normal_from_texture
                                    } else {
                                        normal
                                    };

                                    let amount_of_light = normal_vector.dot(&self.global_light);
                                    let color = Color32::from_gray((amount_of_light * 255.0) as u8);

                                    //let triangle_color = normal_texture;
                                    /* let triangle_color = Color32::from_rgb(
                                        (normal_vector.x * 128.0) as u8,
                                        (normal_vector.y * 128.0) as u8,
                                        (normal_vector.z * 128.0) as u8,
                                        ); */

                                    let triangle_color = Color32::from_rgb(
                                        (diffuse_color.r() as f32 * amount_of_light) as u8,
                                        (diffuse_color.g() as f32 * amount_of_light) as u8,
                                        (diffuse_color.b() as f32 * amount_of_light) as u8,
                                    );

                                    if self.buffer.depth_buffer[w * y as usize + x as usize] > depth
                                    {
                                        unsafe {
                                            // image.pixels.get_unchecked_mut(w * y as usize + x as usize) = triangle_color;
                                            // depth_buffer.get_unchecked_mut(w * y as usize + x as usize) = depth;
                                            if (0..w).contains(&x) && (0..h).contains(&y) {
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

        ui.ctx().tex_manager().write().set(
            self.render_texture_canvas,
            ImageDelta::full(self.buffer.image.clone(), TextureOptions::LINEAR),
        );
    }
}

/*
fn process_model(
    light: &Vector3<f32>,
    image: &mut ColorImage,
    depth_buffer: &mut Vec<f32>,
    points: &[Point3<f32>],
    normals: &[Vector3<f32>],
    triangles: &[(usize, usize, usize)],
    triangle_colors: &[Color32],
    w: usize,
    h: usize,
) {
    for (i, ind) in triangles.iter().enumerate() {
        let a = todo!();
        let b = todo!();
        let c = todo!();
        if (b - a).cross(&(c - a)).z < 0.0 {
            continue;
        }

        let is_valid = |point: &Point3<f32>| {
            //(-1.0..1.0).contains(&point.x)
            //   && (-1.0..1.0).contains(&point.y)
            (-1.0..0.0).contains(&point.z)
        };

        let in_bounds = |point: &Point3<f32>| {
            (-1.0..1.0).contains(&point.x)
                && (-1.0..1.0).contains(&point.y)
                && (-1.0..0.0).contains(&point.z)
        };

        let is_visible_triangle = |p1: &Point3<f32>, p2: &Point3<f32>, p3: &Point3<f32>| {
            in_bounds(p1) || in_bounds(p2) || in_bounds(p3)
        };

        let convert = |point: &Point3<f32>| {
            let x = ((point.x + 1.0) / 2.0 * w as f32) as usize;
            let y = ((point.y + 1.0) / 2.0 * h as f32) as usize;
            (x, y)
        };

        let convertf32 = |point: &Point3<f32>| {
            Point3::new(
                (point.x + 1.0) / 2.0 * w as f32,
                (point.y + 1.0) / 2.0 * h as f32,
                point.z,
            )
        };

        if is_visible_triangle(a, b, c) {
            let vertex_a = Vertex::new(convertf32(a), *na, Color32::RED);
            let vertex_b = Vertex::new(convertf32(b), *nb, Color32::BLUE);
            let vertex_c = Vertex::new(convertf32(c), *nc, Color32::GREEN);
            //let face_normal = (na + nb + nc).normalize();
            let triangle_color = triangle_colors[i];

        }

        let mut draw_line = |p1, p2| {
            if is_valid(p1) && is_valid(p2) {
                let p1 = convert(p1);
                let p2 = convert(p2);
                for (x, y) in Bresenham::new(
                    (p1.0 as isize, p1.1 as isize),
                    (p2.0 as isize, p2.1 as isize),
                ) {
                    //eprintln!("{x}, {y}");
                    image.pixels[w * y as usize + x as usize] = Color32::BLUE;
                }
            }
        };

        //draw_line(a, b);
        //draw_line(b, c);
        //draw_line(c, a);
    }
}
*/
