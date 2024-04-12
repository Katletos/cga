use std::mem::swap;

use egui::Color32;
use nalgebra::{Point2, Point3, Vector3};


pub struct Vertex {
    pub camera_view: Point3<f32>,
    //pub world_pos: Point3<f32>,
    pub norm: Vector3<f32>,
    pub color: (f32, f32, f32),
}

impl Vertex {
    pub fn new(point: Point3<f32>, norm: Vector3<f32>, color: Color32) -> Self {
        let [r, g, b, _a] = color.to_normalized_gamma_f32();
        Vertex {
            camera_view: point,
            norm,
            color: (r, g, b),
        }
    }

    pub fn get_xy(&self) -> (f32, f32) {
        (self.camera_view.x, self.camera_view.y)
    }
}

pub fn rasterize_triangle<F: FnMut(Point2<usize>, (f32, f32, f32), f32)>(
    mut p0: Vertex,
    mut p1: Vertex,
    mut p2: Vertex,
    f: &mut F,
) {
    let (mut x0, mut y0) = p0.get_xy();
    let (mut x1, mut y1) = p1.get_xy();
    let (mut x2, mut y2) = p2.get_xy();

    if (y1, x1) < (y0, x0) {
        swap(&mut x0, &mut x1);
        swap(&mut y0, &mut y1);
        swap(&mut p0, &mut p1);
    }

    if (y2, x2) < (y0, x0) {
        swap(&mut x0, &mut x2);
        swap(&mut y0, &mut y2);
        swap(&mut p0, &mut p2);
    }

    if (y2, x2) < (y1, x1) {
        swap(&mut x1, &mut x2);
        swap(&mut y1, &mut y2);
        swap(&mut p1, &mut p2);
    }

    if y0 == y2 {
        return;
    }

    let shortside = (y1 - y0) * (x2 - x0) < (x1 - x0) * (y2 - y0);
    let mut sides: [_; 2] = [SlopeData::default(); 2];

    sides[(!shortside) as usize] = SlopeData::new(&p0, &p2, (y2 - y0) as usize);

    if y0 < y1 {
        sides[shortside as usize] = SlopeData::new(&p0, &p1, (y1 - y0) as usize);
        for y in (y0 as usize)..(y1 as usize) {
            let [s0, s1] = &mut sides;
            scanline(y, s0, s1, f);
        }
    }

    if y1 < y2 {
        sides[shortside as usize] = SlopeData::new(&p1, &p2, (y2 - y1) as usize);
        for y in (y1 as usize)..(y2 as usize) {
            let [s0, s1] = &mut sides;
            scanline(y, s0, s1, f);
        }
    }
}

fn scanline<F: FnMut(Point2<usize>, (f32, f32, f32), f32)>(y: usize, left: &mut SlopeData, right: &mut SlopeData, f: &mut F) {
    let x = left.get_first() as usize;
    let endx = right.get_first() as usize;
    let num_steps = endx - x;

    let (lr, lg, lb) = left.get_rgb();
    let (rr, rg, rb) = right.get_rgb();
    let r = Slope::new(lr, rr, num_steps);
    let g = Slope::new(lg, rg, num_steps);
    let b = Slope::new(lb, rb, num_steps);
    let depth = Slope::new(left.get_depth(), right.get_depth(), num_steps);
    let mut props = [r, g, b, depth];

    for x in (left.get_first() as usize)..(right.get_first() as usize) {
        f(Point2::new(x, y), (props[0].get(), props[1].get(), props[2].get()), props[3].get());
        for prop in &mut props {
            prop.advance();
        }
    }
    left.advance();
    right.advance();
}

#[derive(Debug, Default, Clone, Copy)]
struct Slope {
    begin: f32,
    step_size: f32,
}

impl Slope {
    fn new(begin: f32, end: f32, num_steps: usize) -> Slope {
        let inv_step = 1.0 / num_steps as f32;
        Slope {
            begin: begin as f32,
            step_size: (end - begin) as f32 * inv_step,
        }
    }

    fn advance(&mut self) {
        self.begin += self.step_size;
    }

    #[inline]
    fn get(self) -> f32 {
        self.begin
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct SlopeData {
    data: [Slope; 5]
}

impl SlopeData {
    fn new(from: &Vertex, to: &Vertex, num_steps: usize) -> Self {
        let slope_x = Slope::new(from.camera_view.x, to.camera_view.x, num_steps);
        let r = Slope::new(from.norm.x, to.norm.x, num_steps);
        let g = Slope::new(from.norm.y, to.norm.y, num_steps);
        let b = Slope::new(from.norm.z, to.norm.z, num_steps);
        let depth = Slope::new(from.camera_view.z, to.camera_view.z, num_steps);

        SlopeData {
            data: [slope_x, r, g, b, depth]
        }
    }

    fn get_first(&self) -> f32 {
        self.data[0].get()
    }

    fn get_rgb(&self) -> (f32, f32, f32){
        (
            self.data[1].get(),
            self.data[2].get(),
            self.data[3].get(),
        )
    }
    
    fn get_depth(&self) -> f32 {
        self.data[4].get()
    }

    fn advance(&mut self) {
        for slope in &mut self.data {
            slope.advance()
        }
    }
}
