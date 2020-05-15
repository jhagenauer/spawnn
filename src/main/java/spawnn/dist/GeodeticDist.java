package spawnn.dist;

import org.geotools.geometry.jts.JTS;
import org.geotools.referencing.GeodeticCalculator;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.TransformException;

import com.vividsolutions.jts.geom.Coordinate;

public class GeodeticDist implements Dist<double[]> {
	private int[] idx;
	GeodeticCalculator gc;

	public GeodeticDist(CoordinateReferenceSystem crs, int[] idx ) {
		this.idx = idx;
		gc = new GeodeticCalculator(crs);
	}

	@Override
	public double dist(double[] a, double[] b) {
		CoordinateReferenceSystem crs = gc.getCoordinateReferenceSystem();
		try {
			gc.setStartingPosition(JTS.toDirectPosition(new Coordinate(a[idx[0]], a[idx[1]]), crs ));
			gc.setDestinationPosition(JTS.toDirectPosition(new Coordinate(b[idx[0]], b[idx[1]]), crs ));
		} catch (TransformException e) {
			e.printStackTrace();
		}
	    return gc.getOrthodromicDistance();
	}
}
