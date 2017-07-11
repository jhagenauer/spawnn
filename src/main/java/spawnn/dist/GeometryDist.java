package spawnn.dist;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.vividsolutions.jts.geom.Geometry;

public class GeometryDist implements Dist<double[]> {
	
	Map<double[],Geometry> m;
	
	public GeometryDist(List<double[]> samples, List<Geometry> geoms) {
		this.m = new HashMap<>();
		for( int i = 0; i < samples.size(); i++ )
			m.put(samples.get(i),geoms.get(i));
	}
	
	@Override
	public double dist(double[] a, double[] b) {
		return m.get(a).distance(m.get(b));
	}
}
