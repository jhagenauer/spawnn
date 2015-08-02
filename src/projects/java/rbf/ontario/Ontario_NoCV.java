package rbf.ontario;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.log4j.Logger;
import org.geotools.data.DataStore;
import org.geotools.data.FeatureSource;
import org.geotools.data.shapefile.ShapefileDataStore;
import org.geotools.factory.CommonFactoryFinder;
import org.geotools.feature.FeatureCollection;
import org.geotools.feature.FeatureIterator;
import org.geotools.geometry.jts.JTS;
import org.geotools.referencing.CRS;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.filter.Filter;
import org.opengis.filter.FilterFactory2;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.rbf.AdaptIncRBF;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.Point;

public class Ontario_NoCV {

	private static Logger log = Logger.getLogger(Ontario_NoCV.class);
	
	public static void main(String[] args) {
		final Random r = new Random();
		
		SpatialDataFrame sd = DataUtils.readShapedata( new File("data/ontario/clipped/ontario_inorg_sel_final.shp"), new int[]{}, true);
		DataUtils.writeCSV("output/ontario.csv", sd.samples, sd.names.toArray(new String[]{}));
				
		log.debug("Building grid...");
		List<double[]> gp = new ArrayList<double[]>(); 
		DataStore dataStore = null;
		try {
			dataStore = new ShapefileDataStore((new File("data/ontario/eco_regions/ger_000b11a_e.shp")).toURI().toURL());
			FeatureSource<SimpleFeatureType, SimpleFeature> fs = dataStore.getFeatureSource(dataStore.getTypeNames()[0]);
			
			CoordinateReferenceSystem sourceCRS = fs.getSchema().getCoordinateReferenceSystem();
			CoordinateReferenceSystem targetCRS =  CRS.decode("EPSG:3724");
			MathTransform transform = CRS.findMathTransform(sourceCRS, targetCRS, true);
											
			FilterFactory2 ff = CommonFactoryFinder.getFilterFactory2(null);
			List<Filter> list = new ArrayList<Filter>();
			list.add( ff.equals(ff.property("ERUID"), ff.literal(3570)) );
			list.add( ff.equals(ff.property("ERUID"), ff.literal(3560)) );
			list.add( ff.equals(ff.property("ERUID"), ff.literal(3580)) );
			list.add( ff.equals(ff.property("ERUID"), ff.literal(3550)) );
			list.add( ff.equals(ff.property("ERUID"), ff.literal(3540)) );
			list.add( ff.equals(ff.property("ERUID"), ff.literal(3530)) );
			
			FeatureCollection fc = fs.getFeatures(ff.or(list) );
			Envelope env = fc.getBounds();
			env = JTS.transform(env, transform);
			
			List<Geometry> geoms = new ArrayList<Geometry>();
			FeatureIterator it = null;
			try {
				it = fc.features();
				while (it.hasNext()) {
					SimpleFeature sf = (SimpleFeature)it.next();
					geoms.add( JTS.transform( (Geometry)sf.getDefaultGeometry(), transform ) );
				}
			} catch (Exception e) {
				e.printStackTrace();
			} finally {
				it.close();
			}	
			
			final int ints = 250;
			GeometryFactory gf = new GeometryFactory();
			for( double x = env.getMinX(); x <= env.getMaxX(); x += env.getWidth()/ints ) {
				for( double y = env.getMinY(); y < env.getMaxY(); y+= env.getHeight()/ints ) {
					
					Point p = gf.createPoint( new Coordinate(x, y));
					for( Geometry geom :geoms ) {
						if( !geom.intersects(p) )
							continue;																		
						gp.add(new double[]{x,y});
					}
				}
			}	
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			dataStore.dispose();
		}
				
		for( int run = 0; run < 1000; run++) {
			log.debug("Run: "+run);
		
			Dist<double[]> dist = new EuclideanDist();
			Map<String,Map<double[],Double>> preds = new HashMap<String,Map<double[],Double>>();
					
			for( int n : new int[]{ 33 } ) // co (14), ni (33), zn (55), al (57), V (51)
				for( int nnSize : new int[]{ 0, 7 } ) {
						
				EuclideanDist dist23 = new EuclideanDist(new int[] { 2, 3 });
				List<double[]> all = new ArrayList<double[]>();
				for (double[] d : sd.samples ) {
								
					List<double[]> nns = new ArrayList<double[]>();
					while( nns.size() < nnSize ) {
						
						double[] nn = null;
						for( double[] d2 : sd.samples ) {
							if( d == d2 || nns.contains(d2) )
								continue;
							
							if( nn == null || dist23.dist(d, d2) < dist23.dist( nn, d)  )
								nn = d2;
						}
						nns.add(nn);
					}
					
					double[] d3 = new double[3+nns.size()];
					d3[0] = d[2]; // x
					d3[1] = d[3]; // y
					d3[2] = d[n]; // n (target)
					for( int i = 0; i < nns.size(); i++ )
						d3[i+3] = nns.get(i)[n];
					all.add( d3 );
				}
							
				int[] ga = new int[]{ 0, 1 };
				Dist<double[]> distGa = new EuclideanDist(ga);
				
				// zScore geo
				double[] meanGeo = new double[ga.length];
				for (double[] s : all)
					for( int i = 0; i < ga.length; i++ )
						meanGeo[i] += s[ga[i]] / all.size();
	
				double sdGeo = 0;
				for (double[] s : all)
					sdGeo += Math.pow(distGa.dist(s, meanGeo), 2);
				sdGeo = Math.sqrt( sdGeo / all.size()-1 );
										
				for (double[] s : all)
					for (int i = 0; i < ga.length; i++ )
						s[ga[i]] = (s[ga[i]] - meanGeo[i]) / sdGeo;
				
				// zScore fa1
				int[] fa1 = new int[all.get(0).length-3];	
				for( int i = 0; i < fa1.length; i++ )
					fa1[i] = i+3;	
				
				double[] meanFa1 = new double[fa1.length];
				for( double[] s : all )
					for( int i = 0; i < fa1.length; i++ )
						meanFa1[i] += s[fa1[i]]/all.size();
										
				double[] sdFa1 = new double[fa1.length];
				for( double[] s : all )
					for( int i = 0; i < fa1.length; i++ )
						sdFa1[i] += Math.pow( meanFa1[i] - s[fa1[i]], 2 );
				for( int i = 0; i < fa1.length; i++ )
					sdFa1[i] = Math.sqrt( sdFa1[i] / (all.size()-1) );
				
				for (double[] s : all)
				for (int i = 0; i < fa1.length; i++ )
					s[fa1[i]] = (s[fa1[i]] - meanFa1[i]) / sdFa1[i];
									
				// zScore fa2			
				int fa2 =2;	
							
				double meanFa2 = 0;
				for( double[] s : all )
					meanFa2 += s[fa2]/all.size();
										
				double sdFa2 = 0;
				for( double[] s : all )
					sdFa2 += Math.pow( meanFa2 - s[fa2], 2 );
				sdFa2 = Math.sqrt( sdFa2 / (all.size()-1) );
				
				for (double[] s : all)
					s[fa2] = (s[fa2] - meanFa2) / sdFa2;
				
				for( final int aMode : new int []{ 1 } ) 
				for( int t_max : new int[]{ 50000 } )
				for( final int ins_per : new int[]{ t_max/10 } )
				for( final double delta : new double[]{ 0.05 } )
				for( final int a_max : new int[]{ 50 } )
				for( final double lrA : new double[]   { 0.05 }  )
				for( final double lrB : new double[]   { 0.001 }  )
				for( final double alpha : new double[] { 0.0004 } )
				for( final double beta : new double[] { 0.5 } )
				for( final double sc : new double[]{ 0.0, 0.1 } )
				for( final int adapt_per : new int[]{ 1000 } ) {
				
				Collections.shuffle(all);
				final List<double[]> samples = new ArrayList<double[]>();
				final List<double[]> desired = new ArrayList<double[]>();
	
				for( double[] d : all ) {
					
					double[] n2 = new double[ga.length+fa1.length];
					for( int i = 0; i < ga.length; i++ )
						n2[i] = d[ga[i]];
					for( int i = 0; i < fa1.length; i++ )
						n2[i+ga.length] = d[fa1[i]];
										
					samples.add(n2);				
					desired.add(new double[] { d[2] });
				}
	
				Map<double[], Double> hidden = new HashMap<double[], Double>();
				while (hidden.size() < 2) {
					double[] d = samples.get(r.nextInt(samples.size()));
					hidden.put(Arrays.copyOf(d, d.length), 1.0);
				}
	
				AdaptIncRBF rbf = new AdaptIncRBF(hidden, lrA, lrB, dist, a_max, sc, alpha, beta, delta, 1);
				for (int t = 0; t < t_max; t++) {
					int idx = r.nextInt(samples.size());
					rbf.train(samples.get(idx), desired.get(idx));
	
					if( t % ins_per == 0 )
						rbf.insert();
					else if( t % adapt_per == 0 ) {				
						if( aMode == 0 )
							rbf.adaptScale( rbf.getTotalError() ); // conventional
						else 
							rbf.adaptScale( Ontario.getRMSE(rbf, samples, desired) );
					}
				}
				
				String desc = "";
				if( sc == 0 ) 
					desc += "Basic";
				else
					desc += "Adaptive";
				
				if( nnSize > 0 )
					desc += " with NBs";
				
				
				//String desc = aMode+","+nnSize+","+lrA+","+lrB+","+ins_per+","+delta+","+a_max+","+sc+","+alpha+","+beta+","+t_max+","+n+","+adapt_per;
							
				List<double[]> response = new ArrayList<double[]>(); 
				for (double[] x : samples)
					response.add(rbf.present(x));
				log.debug(desc+","+rbf.scale+","+Meuse.getRMSE(response,desired)+","+Math.pow(Meuse.getPearson(response,desired), 2)+","+rbf.getNeurons().size() );
							
				for( double[] p : gp ) {		
					
					// get nearest neighbors
					List<double[]> nns = new ArrayList<double[]>();
					while( nns.size() < nnSize ) {
						
						double[] nn = null;
						double minDist = Double.MAX_VALUE;
						
						for( double[] d : sd.samples ) {
							if( nns.contains(d) )
								continue;
							
							double[] pos = new double[]{ d[2], d[3] };						
							if( nn == null || dist.dist(p, pos) < minDist  ) {
								minDist = dist.dist(p, pos);
								nn = d;
							}	
						}
						nns.add(nn);
					}
									
					double[] np = new double[ga.length+fa1.length];
					for( int i = 0; i < ga.length; i++ )
						np[i] = (p[ga[i]] - meanGeo[i])/sdGeo;
												
					for( int i = 0; i < nns.size(); i++ )
						np[i+ga.length] = (nns.get(i)[n] - meanFa1[i]) / sdFa1[i];
					
					double re = rbf.present( np )[0]; 
					re = re * sdFa2 + meanFa2; // reconstruct prediction
					
					if( !preds.containsKey(desc) )
						preds.put( desc, new HashMap<double[],Double>() );
					preds.get(desc).put(p, re);
				}
			}
			
				FileWriter fw = null;
				try {
					fw = new FileWriter("output/prediction_"+run+".csv");
					
					List<String> keys = new ArrayList<String>(preds.keySet() );
					
					fw.write("x,y");
					for( String k : keys )
						fw.write("," + k);
					fw.write("\n");
					
					for( double[] p : gp ) { 
						fw.write( p[0]+","+p[1] );
						for( String k : keys )
							fw.write(","+preds.get(k).get(p) );
						fw.write("\n");
					}
											
					fw.close();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
	}
}
