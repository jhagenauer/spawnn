package spawnn.dist;

import com.vividsolutions.jts.geom.Geometry;

public class GeometryDist implements Dist<Geometry> {
	
	@Override
	public double dist(Geometry a, Geometry b) {
		return a.distance(b);
	}
}
