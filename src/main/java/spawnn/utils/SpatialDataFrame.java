package spawnn.utils;

import java.util.List;

import org.opengis.referencing.crs.CoordinateReferenceSystem;

import com.vividsolutions.jts.geom.Geometry;

public class SpatialDataFrame extends DataFrame {
	
	public List<Geometry> geoms;
	public CoordinateReferenceSystem crs;
}
