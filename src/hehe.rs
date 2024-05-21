
        let point = Point3::new(0.0, 0.0, 0.0);
        let point1 = Point3::new(1.0, 0.0, 0.0);
        let point2 = Point3::new(0.0, 1.0, 0.0);
        let point3 = Point3::new(0.0, 0.0, 1.0);

        dbg!(model_view.transform_point(&point));
        let point_coords = to_screen_space(point);
        let point_coords1 = to_screen_space(point1);
        let point_coords2 = to_screen_space(point2);
        let point_coords3 = to_screen_space(point3);
        dbg!(point_coords);

        if (-1.0..1.0).contains(&point_coords.z) &&
            (-1.0..1.0).contains(&point_coords.x) &&
            (-1.0..1.0).contains(&point_coords.y)
        {
            let z = (point_coords.z + 1.0) / 2.0;
            let radius = 10.0 / z;
            dbg!(radius);
            painter.circle(
                pos2(
                    (point_coords.x + 1.0) / 2.0 * wh[0] as f32,
                    (1.0 - point_coords.y) / 2.0 * wh[1] as f32,
                ),
                radius,
                Color32::WHITE,
                (1.0, Color32::YELLOW),
            );
        }

        if (-1.0..1.0).contains(&point_coords1.z) &&
            (-1.0..1.0).contains(&point_coords1.x) &&
            (-1.0..1.0).contains(&point_coords1.y)
        {
            let z = (point_coords1.z + 1.0) / 2.0;
            let radius = 10.0 / z;
            dbg!(radius);
            painter.circle(
                pos2(
                    (point_coords1.x + 1.0) / 2.0 * wh[0] as f32,
                    (1.0 - point_coords1.y) / 2.0 * wh[1] as f32,
                ),
                radius,
                Color32::GREEN,
                (1.0, Color32::YELLOW),
            );
        }

        if (-1.0..1.0).contains(&point_coords2.z) &&
            (-1.0..1.0).contains(&point_coords2.x) &&
            (-1.0..1.0).contains(&point_coords2.y)
        {
            let z = (point_coords2.z + 1.0) / 2.0;
            let radius = 10.0 / z;
            dbg!(radius);
            painter.circle(
                pos2(
                    (point_coords2.x + 1.0) / 2.0 * wh[0] as f32,
                    (1.0 - point_coords2.y) / 2.0 * wh[1] as f32,
                ),
                radius,
                Color32::RED,
                (1.0, Color32::YELLOW),
            );
        }

        if (-1.0..1.0).contains(&point_coords3.z) &&
            (-1.0..1.0).contains(&point_coords3.x) &&
            (-1.0..1.0).contains(&point_coords3.y)
        {
            let z = (point_coords3.z + 1.0) / 2.0;
            let radius = 10.0 / z;
            dbg!(radius);
            painter.circle(
                pos2(
                    (point_coords3.x + 1.0) / 2.0 * wh[0] as f32,
                    (1.0 - point_coords3.y) / 2.0 * wh[1] as f32,
                ),
                radius,
                Color32::BLUE,
                (1.0, Color32::YELLOW),
            );
        }
