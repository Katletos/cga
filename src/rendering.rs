use nalgebra::{Isometry3, Matrix4, Point3, Vector3};

pub struct Camera {
    pub eye: Point3<f32>,
    pub target: Point3::<f32>,
    pub up: Vector3::<f32>,
    pub look_at: Matrix4<f32>,
    pub look_at2: Isometry3<f32>,
}

impl Camera {
    pub fn new(distance: f32, angle: f32, height: f32) -> Self {
        let x = angle.sin() * distance;
        let y = angle.cos() * distance;
        let eye = Point3::new(x, height, y);

        let target = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, -1.0, 0.0);
        Camera {
            eye,
            target,
            up,
            look_at: Matrix4::look_at_rh(&eye, &target, &up),
            look_at2: Isometry3::look_at_rh(&eye, &target, &up),
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        let eye = Point3::new(100.0, 100.0, 100.0);
        let target = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 0.0, 1.0);

        Camera {
            eye,
            target,
            up,
            look_at: Matrix4::look_at_rh(&eye, &target, &up),
            look_at2: Isometry3::look_at_rh(&eye, &target, &up),
        }
    }
}
