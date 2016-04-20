package hstsom.austria;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.Collections;
import java.util.Map;

import org.geotools.data.DataStore;
import org.geotools.data.DataUtilities;
import org.geotools.data.FeatureStore;
import org.geotools.data.FileDataStoreFactorySpi;
import org.geotools.data.shapefile.ShapefileDataStore;
import org.geotools.data.shapefile.ShapefileDataStoreFactory;
import org.geotools.factory.CommonFactoryFinder;
import org.geotools.factory.GeoTools;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.FeatureCollection;
import org.geotools.feature.FeatureIterator;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.filter.FilterFactory2;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.MultiPolygon;

public class AuMuDataBuilder {

	public static void main(String[] args) {
			FilterFactory2 ff = CommonFactoryFinder.getFilterFactory2(GeoTools.getDefaultHints());
			
			FeatureCollection<SimpleFeatureType,SimpleFeature> trend = null;
	    	try {
	    		{
	    			DataStore dataStore = new ShapefileDataStore( (new File("data/sps/municipalitiesAustria/trend.shp")).toURI().toURL());
	    			trend = DataUtilities.collection( dataStore.getFeatureSource(dataStore.getTypeNames()[0]).getFeatures() );
	    		}
	    	} catch(Exception e ) {
	    		e.printStackTrace();
	    		System.exit(1);
	    	}
	    	
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("trend");
			typeBuilder.add("YEAR",Integer.class);
			typeBuilder.add("X",Integer.class);
			typeBuilder.add("Y",Integer.class);
			typeBuilder.add("ORIGID",String.class);
			typeBuilder.add("NAME",String.class);
			typeBuilder.add("AREA",Double.class);
			typeBuilder.add("BEV",Integer.class);
			typeBuilder.add("BEVDENS",Double.class);
			typeBuilder.add("LNBVDENS",Double.class);
			typeBuilder.add("BES",Integer.class);
			typeBuilder.add("BESRATE",Double.class);
			typeBuilder.add("AUSP",Integer.class);
			typeBuilder.add("AUSPRATE",Double.class);
			typeBuilder.add("ALLE",Integer.class);
			typeBuilder.add("ALLEDENS",Double.class);
			typeBuilder.add("LNALLDNS",Double.class);
			
			typeBuilder.add("the_geom",MultiPolygon.class);
			
			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder( typeBuilder.buildFeatureType() );
	    	
			DefaultFeatureCollection fc = new DefaultFeatureCollection();
	    	    	
	    	FeatureIterator it1 = trend.features();
			try {
				while( it1.hasNext() ) {
					SimpleFeature sf = (SimpleFeature) it1.next();
					Geometry geom = ((Geometry)sf.getDefaultGeometry()).buffer(0);
					
					String id = sf.getAttribute("ID").toString();
					int x = Integer.parseInt( sf.getAttribute("X").toString() );
					int y = Integer.parseInt( sf.getAttribute("Y").toString() );
					String name = sf.getAttribute("NAME").toString();
					double area = Double.parseDouble( sf.getAttribute("AREA").toString() )/1000000.0;
					
					for( String year : new String[]{"1961","1971","1981","1991","2001"} ) {
						String s = year.substring(2); 
						int bev = Integer.parseInt( sf.getAttribute("BEV"+s).toString() );
						int bes = Integer.parseInt( sf.getAttribute("BES"+s).toString() );
						int ausp = Integer.parseInt( sf.getAttribute("AUS"+s).toString() );
						
						
						int alle = -1;
						if( s.equals("61") )
							alle = Integer.parseInt( sf.getAttribute("ALLE64").toString() );
						else if( s.equals("71") )
							alle = Integer.parseInt( sf.getAttribute("ALLE73").toString() );
						else
							alle = Integer.parseInt( sf.getAttribute("ALLE"+s).toString() );
						
						// build feature
						featureBuilder.set( "YEAR", year);								
						featureBuilder.set( "X", x);									
						featureBuilder.set( "Y", y);									
						featureBuilder.set( "ORIGID", id);
						featureBuilder.set( "NAME", name);
						featureBuilder.set( "AREA", area);
						featureBuilder.set( "BEV", bev);								
						featureBuilder.set( "BEVDENS", (double)bev/area);
						featureBuilder.set( "LNBVDENS", Math.log( (double)bev/area));		
						featureBuilder.set( "BES", bes);
						featureBuilder.set( "BESRATE", (double)bes/bev);				
						featureBuilder.set( "AUSP", ausp);
						featureBuilder.set( "AUSPRATE", (double)ausp/bes);
						featureBuilder.set( "ALLE", alle);								
						featureBuilder.set( "ALLEDENS", (double)alle/area);
						featureBuilder.set("LNALLDNS", Math.log( (double)alle/area ) );
						featureBuilder.set("the_geom", geom);
						
						fc.add( featureBuilder.buildFeature(""+fc.size() ) );
					}
				}
			} finally {
				if( it1 != null ) {
					it1.close(); // YOU MUST CLOSE THE ITERATOR!
				}
			}
			
			
			// store
			//write shape 
			try {
				// store shape file
				File out = new File("/home/julian/tmp/munaus_3.shp");
				Map map = Collections.singletonMap("url", out.toURI().toURL());
		        FileDataStoreFactorySpi factory = new ShapefileDataStoreFactory();
				DataStore myData = factory.createNewDataStore(map);
				myData.createSchema((SimpleFeatureType) fc.getSchema());
				String n = fc.getSchema().getName().getLocalPart();
				FeatureStore<SimpleFeatureType, SimpleFeature> store = (FeatureStore<SimpleFeatureType, SimpleFeature>) myData.getFeatureSource(n);
				store.addFeatures(fc);
	
			} catch (MalformedURLException ex) {
				ex.printStackTrace();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
			
			System.out.println("done");
	
	}
}
