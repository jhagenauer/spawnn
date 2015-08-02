package cng_houseprice;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;
import org.geotools.data.DataStore;
import org.geotools.data.FeatureSource;
import org.geotools.data.shapefile.ShapefileDataStore;
import org.geotools.feature.FeatureIterator;
import org.geotools.geometry.jts.JTS;
import org.geotools.referencing.CRS;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;

import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.Point;

public class PrepareHousingData {

	private static Logger log = Logger.getLogger(PrepareHousingData.class);

	public static void main(String[] args) {
		GeometryFactory gf = new GeometryFactory();
		
		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/econ_toolbox/house.csv"), new int[] { 15 }, true);
		
		// prepare census tract data
		Map<Geometry,double[]> censusData = new HashMap<Geometry,double[]>();
		{
		DataFrame dfA = DataUtils.readDataFrameFromCSV(new File("data/census/tracts/selected.csv"), new int[]{}, true); // population
		
		DataStore dataStore = null;
		try {
			dataStore = new ShapefileDataStore((new File("data/census/tracts/tr39_d00.shp")).toURI().toURL());
			FeatureSource<SimpleFeatureType, SimpleFeature> featureSource = dataStore.getFeatureSource(dataStore.getTypeNames()[0]);
			CoordinateReferenceSystem crs = featureSource.getSchema().getCoordinateReferenceSystem();
			MathTransform mt = CRS.findMathTransform( crs == null ? CRS.decode("EPSG:4326") : crs, CRS.decode("EPSG:26917") );

			FeatureIterator<SimpleFeature> it = featureSource.getFeatures().features();
			try {
				while (it.hasNext()) {
					SimpleFeature feature = it.next();
					String id = feature.getAttribute("STATE").toString()+feature.getAttribute("COUNTY").toString()+feature.getAttribute("TRACT").toString();
					while( id.length() < 11 ) {
						id += "0";
					}
										
					for( double[] d : dfA.samples ) {
						String dId = Long.toString( (long)d[0] );
						if( dId.equals(id) ) {
							// convert to utm
							Geometry g = (Geometry)feature.getDefaultGeometry();					
							censusData.put( g, new double[]{ d[1]/JTS.transform(g, mt).getArea(), d[2]/d[1], d[5], d[6] } );
						} 
					}
				}
			}catch( Exception e ) {
				e.printStackTrace();
			} finally {
				if (it != null) 
					it.close();
				dataStore.dispose();
			}
			
			log.debug(censusData.size());
			
		} catch(Exception e ) {
			e.printStackTrace();
		}
		}
				
		// bivant
		int[] coords = new int[] { 15, 16 };
		
		List<double[]> all = new ArrayList<double[]>();
		for (double[] d : df.samples) {
			double lon = d[coords[0]];
			double lat = d[coords[1]];
			double age = 1999-d[1];
			double ageSquared = Math.pow(age, 2);
			double ageCubed = Math.pow(age, 3);
			double sold93 = d[17];
			double sold94 = d[18];
			double sold95 = d[19];
			double sold96 = d[20];
			double sold97 = d[21];
			double sold98 = d[22];
			double lnLotSize = Math.log(d[13]);
			double lnTLA = Math.log(d[3]);
			double rooms = d[12];
			double beds = d[5];
			double baths = d[6];
			double lnPrice = Math.log(d[0]);
			double sale_date = d[14];
			double grgType = d[10];
			double grgPresent = d[10] > 0 ? 1 : 0;
			double stories = d[2];
			double lnGrgSqft = Math.log( 2.08 + d[11] );
			double lnFrontage = Math.log( 1+d[8] );
			double wallcode = d[4];
			
			Point p = gf.createPoint(new Coordinate(lon, lat));
			Geometry closest = null;
			for( Geometry g : censusData.keySet() ) {
				if( closest == null || g.distance(p) < closest.distance(p))
					closest = g;
			}
			double[] c = censusData.get(closest);
			double[] nd = { lon,lat, 
					age, 
				//	ageSquared, 
					lnTLA, 
				//	beds, 
					rooms, 
					lnLotSize, 
					grgPresent,
					
				//	stories,
				//	lnGrgSqft,
				//	lnFrontage,
				//	wallcode,
										
				//	c[0], 
					Math.log(1-c[1]), 
					c[2], 
				//	c[3],
					
					lnPrice }; // marco
			
			all.add(nd);
		}
		
		String[] names = {"lon","lat", 
				"age",
			//	"ageSquared", 
				"lnTLA", 
			//	"beds", 
				"rooms", "lnLotSize", "grgPresent",
				
			//	"stories",
			//	"lnGrgSqft",
			//	"lnFrontage",
			//	"wallcode",
								
			//	"popDens",
				"lnNoneWhite",
				"meanHHEarnings",
			//	"bachelorPCT",
				
				"lnPrice"};
				
		final int[] fa1 = new int[all.get(0).length-3];
		for( int i = 0; i < fa1.length; i++ )
			fa1[i] = i + 2;
		int fa2 = all.get(0).length - 1;
		
		/*for( int i : fa1 ) {
			DataUtils.zScoreColumn(all, i);
			//DataUtils.normalizeColumn(all, i);
		}*/
				
		//DataUtils.zScoreColumn(all, fa2);
		
		//Collections.shuffle(all);
		//all = all.subList(0, 7000);
		
		DataUtils.writeCSV("output/house_sample.csv", all, names );	
		
		/*log.debug("Building distance map...");
		Map<double[], Map<double[], Double>> dm = GeoUtils.getInverseDistanceMatrix(all, new EuclideanDist(new int[]{0,1}), 2, true);
		log.debug("done.");
		
		for( int i : fa1 )
			log.debug(names[i]+","+GeoUtils.getMoransI( dm, i));*/		
	}
}
