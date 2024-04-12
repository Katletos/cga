use crate::bresenham::{Bresenham, Point};
use crate::rendering::Camera;
use crate::scanline::{rasterize_triangle, Vertex};
use itertools::Itertools;
use rayon::prelude::*;

use egui::{
    epaint::ImageDelta, include_image, pos2, vec2, Color32, ColorImage, Direction, Frame, Image,
    Margin, Rect, Sense, TextureId, TextureOptions, Vec2,
};

use nalgebra::{wrap, Isometry3, Matrix4, Perspective3, Point2, Point3, Vector3};
use nalgebra_glm;
use nalgebra_glm::Vec3;
use obj::{load_obj, Obj};
use rand::{thread_rng, Rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};
use std::cmp::Ordering;
use std::fs::File;
use std::io::BufReader;
use std::ops::RangeBounds;
use std::time::Instant;
use std::vec;
use wavefront_obj::obj::ObjSet;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
// #[derive(serde::Deserialize, serde::Serialize, Default)]
// #[serde(default)]
pub struct DickDrawingApp {
    texture: TextureId,
    depth_buffer: Vec<f32>,
    obj: Option<Obj>,
    obj2: Option<ObjSet>,
    distance: f32,
    angle: f32,
    height: f32,
    rasterize: bool,
    points: Vec<Point2<f32>>,
    vertices: Vec<usize>,
    colors: Vec<Color32>,
    model: wavefront::Obj,
    model_points: Vec<Vertex>,
    model_triangles: Vec<(usize, usize, usize)>,
    triangle_colors: Vec<Color32>,
    points_buf: Vec<Point3<f32>>,
    normals_buf: Vec<Vector3<f32>>,
    prev_light: Vector3<f32>,
    light: Vector3<f32>,
}

impl DickDrawingApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        let texture = cc.egui_ctx.tex_manager().write().alloc(
            "app".parse().unwrap(),
            ColorImage::default().into(),
            TextureOptions::NEAREST,
        );

        //let model = wavefront::Obj::from_file("aboba/marcus.obj").unwrap();
        let model = wavefront::Obj::from_file("teapot.obj").unwrap();
        //let model = wavefront::Obj::from_file("Napoleon.obj").unwrap();
        //let model = wavefront::Obj::from_file("aboba/model_2.obj").unwrap();

        let mut rng = thread_rng();

        let mut points = Vec::new();
        let mut test_rasterizer_points = Vec::new();

        let mut colors = Vec::new();

        for i in 0..500 {
            let x: f64 = rng.gen::<f64>() * 2.0f64 - 1.0f64;
            let y: f64 = rng.gen::<f64>() * 2.0f64 - 1.0f64;
            points.push(delaunator::Point { x, y });
            test_rasterizer_points.push(Point2::new(x as f32, y as f32));
        }

        for i in 0..10000 {
            colors.push(Color32::from_rgb(rng.gen(), rng.gen(), rng.gen()));
        }

        let triangles = delaunator::triangulate(&points);
        let vertices = triangles.triangles;

        let mut model_points = Vec::new();
        for (kek) in model.positions().iter().zip_longest(model.normals()) {
            match kek {
                itertools::EitherOrBoth::Both(v, n) => {
                    let vertex = v;
                    let normal = *n;
                    model_points.push(Vertex::new(Point3::from(*vertex), Vector3::from(normal), Color32::BLACK));
                },
                itertools::EitherOrBoth::Left(v) => {
                    let vertex = v;
                    let normal = [0.0; 3];
                    model_points.push(Vertex::new(Point3::from(*vertex), Vector3::from(normal), Color32::BLACK));
                },
                itertools::EitherOrBoth::Right(_) => {

                },
            }
        }

        let mut model_triangles = Vec::new();
        let mut triangle_colors = Vec::new();


        for [a, b, c] in model.triangles() {
            let (a, b, c) = (a.position_index(), b.position_index(), c.position_index());
            //let (an, bn, cn) = (a.position_index(), b.position_index(), c.position_index());
            model_triangles.push((a, b, c));
        }

        for (a, b, c) in &model_triangles {
            let (a, b, c) = (
                model_points[*a].camera_view,
                model_points[*b].camera_view,
                model_points[*c].camera_view);
            let ab = b - a;
            let ac = c - a;
            let normal = ab.cross(&ac).normalize();
            let light = Vector3::new(0.5, 0.5, 0.5).normalize();
            let amount_of_light = normal.dot(&light);
            let color = Color32::from_gray((amount_of_light * 255.0) as u8);
            triangle_colors.push(color);
        }

        DickDrawingApp {
            triangle_colors,
            texture,
            //obj: Some(obj),
            obj: None,
            obj2: None,
            model,
            //obj2: Some(obj2),
            distance: 3.0,
            angle: 0.0,
            height: 100.0f32,
            rasterize: false,
            light: Vector3::new(0.5, 0.5, 0.5),
            prev_light: Vector3::new(0.5, 0.5, 0.5),

            points: test_rasterizer_points,
            vertices,
            colors,
            depth_buffer: Vec::new(),
            model_points,
            model_triangles,

            points_buf: Vec::new(),
            normals_buf: Vec::new(),
        }
    }
}

impl eframe::App for DickDrawingApp {
    /// Called by the frame work to save state before shutdown.
    /* fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    } */

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default()
            .frame(Frame::none())
            .show(ctx, |ui| {
                let camera = Camera::new(self.distance, self.angle.to_radians(), self.height);
                let far = 10.0;
                let near = far / 100.0;
                let perspective = Perspective3::new(16.0 / 9.0, 60.0f32.to_radians(), 1.0, 1000.0);

                let start = Instant::now();
                self.points_buf.clear();
                self.normals_buf.clear();
                let model = Isometry3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::y() * std::f32::consts::FRAC_2_PI);
                let model_view_projection = camera.look_at2 * model;

                for vertex in &self.model_points {
                    //let a = camera.look_at.transform_point(&Point3::from(*vertex));
                    //self.points_buf.push(perspective.unproject_point(&a));
                    let point = vertex.camera_view;
                    let normal = vertex.norm;
                    let to_push = perspective.unproject_point(&model_view_projection.transform_point(&point));
                    let to_push_normal = model * normal;
                    self.points_buf.push(to_push);
                    self.normals_buf.push(to_push_normal);
                }

                if self.prev_light != self.light {
                    self.prev_light = self.light;
                    self.triangle_colors.clear();
                    for (a, b, c) in &self.model_triangles {
                        let (a, b, c) = (
                            self.model_points[*a].camera_view,
                            self.model_points[*b].camera_view,
                            self.model_points[*c].camera_view);
                        let ab = b - a;
                        let ac = c - a;
                        let normal = ab.cross(&ac).normalize();
                        let amount_of_light = normal.dot(&self.light);
                        let color = Color32::from_gray((amount_of_light * 255.0) as u8);
                        self.triangle_colors.push(color);
                    }
                }


                dbg!("Point transform", start.elapsed());
                let w = ui.clip_rect().width() as usize;
                let h = ui.clip_rect().height() as usize;

                let wh = [w, h];
                dbg!(wh);

                let mut image = ColorImage::new(wh, Color32::BLACK);
                let mut image2 = ColorImage::new(wh, Color32::BLACK);
                self.depth_buffer.fill(f32::INFINITY);

                if self.depth_buffer.len() < image.pixels.len() {
                    self.depth_buffer.resize(image.pixels.len(), f32::INFINITY);
                }
                let mut depth_buf2 = vec![f32::INFINITY; image.pixels.len()];

                rayon::scope(|s| {
                    s.spawn(|_| {
                        process_model(
                            &self.light,
                            &mut image,
                            &mut self.depth_buffer,
                            &self.points_buf,
                            &self.normals_buf,
                            &self.model_triangles[0..self.model_triangles.len() / 2],
                            &self.triangle_colors[0..self.model_triangles.len() / 2],
                            w,
                            h,
                        );
                    });
                    s.spawn(|_| {
                        process_model(
                            &self.light,
                            &mut image2,
                            &mut depth_buf2,
                            &self.points_buf,
                            &self.normals_buf,
                            &self.model_triangles[self.model_triangles.len() / 2..],
                            &self.triangle_colors[self.model_triangles.len() / 2..],
                            w,
                            h,
                        );
                    });
                });

                (
                    &mut image.pixels,
                    &mut self.depth_buffer,
                    &image2.pixels,
                    &depth_buf2,
                )
                    .into_par_iter()
                    .map(|(p1, d1, p2, d2)| {
                        if d2 < d1 {
                            *p1 = *p2;
                        }
                    })
                    .count();

                let mut cursor_x = 0.0;
                let mut cursor_y = 0.0;

                if let Some(pos) = ui.input(|x| x.pointer.hover_pos()) {
                    cursor_x = pos.x;
                    cursor_y = pos.y;
                }
                dbg!("Triangle render", start.elapsed());

                egui::Window::new("Settings").show(ctx, |ui| {
                    ui.add(egui::Slider::new(&mut self.distance, 0.0..=10000.0).text("Distance"));
                    ui.add(egui::Slider::new(&mut self.angle, 0.0..=360.0).text("Angle"));
                    ui.add(egui::Slider::new(&mut self.height, -1000.0..=1000.0).text("Height"));

                    if ui
                        .button(format!("Rasterize: {}", self.rasterize))
                        .clicked()
                    {
                        self.rasterize = !self.rasterize;
                    }

                    ui.add(egui::Slider::new(&mut self.light.x, -1.0..=1.0).text("Light X"));
                    ui.add(egui::Slider::new(&mut self.light.y, -1.0..=1.0).text("Light Y"));
                    ui.add(egui::Slider::new(&mut self.light.z, -1.0..=1.0).text("Light Z"));
                });


                ui.ctx().tex_manager().write().set(
                    self.texture,
                    ImageDelta::full(image, TextureOptions::LINEAR),
                );

                let (response, painter) =
                    ui.allocate_painter(ui.ctx().screen_rect().size(), Sense::drag());

                painter.image(
                    self.texture,
                    painter.clip_rect(),
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
                dbg!("Frame finish", start.elapsed());
            });
    }
}

fn load_obj_from_file(filename: &str) -> Obj {
    let file = File::open(filename).expect("Can't open file");
    let input = BufReader::new(file);
    load_obj(input).expect("Can't load obj")
}

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
        let (ai, bi, ci) = ind;
        let a = &points[*ai];
        let b = &points[*bi];
        let c = &points[*ci];
        let na = &normals[*ai];
        let nb = &normals[*bi];
        let nc = &normals[*ci];

        if (b - a).cross(&(c - a)).z < 0.0 {
            continue;
        }

        let is_valid = |point: &Point3<f32>| {
            (-1.0..1.0).contains(&point.x)
                && (-1.0..1.0).contains(&point.y)
                && (-1.0..0.0).contains(&point.z)
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

        if is_valid(a) && is_valid(b) && is_valid(c) {
            let vertex_a = Vertex::new(convertf32(a), *na, Color32::RED);
            let vertex_b = Vertex::new(convertf32(b), *nb, Color32::BLUE);
            let vertex_c = Vertex::new(convertf32(c), *nc, Color32::GREEN);
            let face_normal = (na + nb + nc).normalize();
            let triangle_color = triangle_colors[i];

            rasterize_triangle(vertex_a, vertex_b, vertex_c, &mut |point, rgb, depth| {
                let x = point.x;
                let y = point.y;
                if (0..w).contains(&x) && (0..h).contains(&y) {
                    /*let color = Color32::from_rgb(
                        (rgb.0 * 255.0) as u8,
                        (rgb.1 * 255.0) as u8,
                        (rgb.2 * 255.0) as u8,
                    );*/
                    //let color = triangle_color;
                    //let mut normal = Vector3::new(rgb.0, rgb.1, rgb.2);

                    //let amount_of_light = face_normal.dot(&light);
                    //let color = Color32::from_gray((amount_of_light * 255.0) as u8);

                    if depth_buffer[w * y as usize + x as usize] > depth {
                        unsafe {
                            *image.pixels.get_unchecked_mut(w * y as usize + x as usize) = triangle_color;
                            *depth_buffer.get_unchecked_mut(w * y as usize + x as usize) = depth;
                        }
                    };
                }
            });
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

        draw_line(a, b);
        draw_line(b, c);
        draw_line(c, a);
    }
}
