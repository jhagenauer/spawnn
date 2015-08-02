package spawnn.utils;

import java.util.List;

import org.apache.log4j.Logger;
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import com.vividsolutions.jts.geom.Geometry;

public class SpatialDataFrame extends DataFrame {
	
	private static Logger log = Logger.getLogger(SpatialDataFrame.class);

	public List<Geometry> geoms;
	public CoordinateReferenceSystem crs;
}
