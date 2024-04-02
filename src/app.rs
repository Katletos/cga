use crate::bresenham::{Bresenham, Point};
use egui::{
    epaint::ImageDelta, include_image, pos2, vec2, Color32, ColorImage, Direction, Frame, Image,
    Margin, Rect, Sense, TextureId, TextureOptions, Vec2,
};
use nalgebra::{Matrix4, Point3, Vector3};
use nalgebra_glm;
use nalgebra_glm::Vec3;
use obj::Obj;
use std::vec;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
//#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct DickDrawingApp {
    texture: TextureId,
    obj: Obj,
}

impl DickDrawingApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>, obj: Obj) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        let texture = cc.egui_ctx.tex_manager().write().alloc(
            "app".parse().unwrap(),
            ColorImage::default().into(),
            TextureOptions::NEAREST,
        );

        Self { texture, obj }
    }
}

impl eframe::App for DickDrawingApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default()
            .frame(Frame::none())
            .show(ctx, |ui| {
                let eye = Point3::<f32>::new(0.0, 0.0, 0.0);
                let target = Point3::<f32>::new(1.0, 1.0, 1.0);
                let up = Vector3::<f32>::new(0.0, 0.0, 1.0);
                let look_at = Matrix4::face_towards(&eye, &target, &up);

                let mut point_in_view_space = Vec::with_capacity(self.obj.vertices.len());
                for vertex in self.obj.vertices.to_vec() {
                    let a = look_at.transform_vector(&Vector3::from(vertex.normal));
                    point_in_view_space.push(a);
                }
                dbg!(point_in_view_space);

                let w = ui.clip_rect().width() as usize;
                let h = ui.clip_rect().height() as usize;
                let wh = [w, h];
                let mut image = ColorImage::new(wh, Color32::YELLOW);
                let mut cursor_x = 0.0;
                let mut cursor_y = 0.0;

                if let Some(pos) = ui.input(|x| x.pointer.hover_pos()) {
                    cursor_x = pos.x;
                    cursor_y = pos.y;
                }

                for (x, y) in Bresenham::new((0, 0), (cursor_x as isize, cursor_y as isize)) {
                    image.pixels[w * y as usize + x as usize] = Color32::BLACK;
                }

                ui.ctx().tex_manager().write().set(
                    self.texture,
                    ImageDelta::full(image, TextureOptions::NEAREST),
                );

                let (response, painter) =
                    ui.allocate_painter(ui.ctx().screen_rect().size(), Sense::drag());
                painter.image(
                    self.texture,
                    painter.clip_rect(),
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
            });
        dbg!("asdasdadsa");
    }
}
