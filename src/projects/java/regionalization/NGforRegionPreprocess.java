package regionalization;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;
import spawnn.utils.RegionUtils;

public class NGforRegionPreprocess {
	private static Logger log = Logger.getLogger(NGforRegionPreprocess.class);
	
	public static void main(String[] args) {
			
		Random r = new Random();
		int T_MAX = 100000;
		
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/redcap/Election/election2004.shp"));
		List<double[]> tmpSamples = DataUtils.readSamplesFromShapeFile(new File("data/redcap/Election/election2004.shp"), new int[]{}, false);
				
		List<double[]> samples = new ArrayList<double[]>();
		for( int i = 0; i < tmpSamples.size(); i++ ) {
			double[] d = tmpSamples.get(i);
			double[] nd = Arrays.copyOf(d, d.length+2);
			Point centroid = geoms.get(i).getCentroid();
			nd[nd.length-2] = centroid.getX();
			nd[nd.length-1] = centroid.getY();
			samples.add(nd);
		}
		
		int vLength = samples.get(0).length;
		int FIPS = 1;
		int[] fa = new int[]{ 7 };
		int[] ga =new int[]{vLength-2,vLength-1};
			
		for( int i : fa)
			DataUtils.zScoreColumn(samples, i );
		
		log.debug("build contiguity map");
		Map<double[],Set<double[]>> cm = new HashMap<double[],Set<double[]>>();
		try {
			BufferedReader br = new BufferedReader(new FileReader("data/redcap/Election/election2004_Queen.ctg"));
			String line = br.readLine();
			while( ( line = br.readLine() ) != null ) {
				String[] split = line.split(",");
				int a = Integer.parseInt(split[0]);
				int b = Integer.parseInt(split[1]);
				
				double[] da = samples.get(a), db = samples.get(b);
				
				if( !cm.containsKey(da) )
					cm.put( da, new HashSet<double[]>() );
							
				assert da != null;
				assert db != null;
				
				cm.get(da).add(db);
			}
		} catch( Exception e ) {
			e.printStackTrace();
		}
		log.debug("done");
		
		
		
		Dist eDist = new EuclideanDist();
		Dist fDist = new EuclideanDist(fa);
		Dist geoDist = new EuclideanDist(ga );
				
		for( int n = 6; n <= 6; n++ ) {
			log.debug("neurons: "+n);
			for( int gns = 1; gns <= n; gns++ ) {
				log.debug("GNS: "+gns);
				
				// cng
				Sorter bmuGetter = new KangasSorter( geoDist, fDist, gns);
				NG ng = new NG(n, n/2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter );
				for (int t = 0; t < T_MAX; t++) {
					double[] x = samples.get(r.nextInt(samples.size() ) );
					ng.train( (double)t/T_MAX, x );
				}
				
				Map<double[],Set<double[]>> cluster = new HashMap<double[],Set<double[]>>();
				for( double[] w : ng.getNeurons() )
					cluster.put( w, new HashSet<double[]>() );
				for( double[] d : samples ) {
					bmuGetter.sort(d, ng.getNeurons() );
					double[] bmu = ng.getNeurons().get(0);
					cluster.get(bmu).add(d);
				}
				
				// get subcluster
				Set<Set<double[]>> subcluster = new HashSet<Set<double[]>>();
				for( Set<double[]> c : cluster.values() ) 
					subcluster.addAll( RegionUtils.getAllContiguousSubcluster(cm, c));
				log.debug("subcluster: "+subcluster.size());
				
				// store subcluster
				GeometryFactory gf = new GeometryFactory();
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("subcluster");
				for( int i : fa )
				typeBuilder.add("fa_"+i, Double.class);
				typeBuilder.add("fips_sum", String.class);
				typeBuilder.add("the_geom",Polygon.class);
				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder( typeBuilder.buildFeatureType() );
				
				DefaultFeatureCollection fc = new DefaultFeatureCollection();
				
				Map<double[],Set<double[]>> oMap = new HashMap<double[],Set<double[]>>();
				
				List<double[]> nSamples = new ArrayList<double[]>();
				List<Geometry> nGeoms = new ArrayList<Geometry>();
				List<double[]> oSamples = DataUtils.readSamplesFromShapeFile(new File("data/redcap/Election/election2004.shp"), new int[]{}, false);
				
				for( Set<double[]> c : subcluster ) {
										
					Geometry[] gs = new Geometry[c.size()];
					double[] sfa = new double[fa.length];
					String sum_fips = "";
					int i = 0;
					for( double[] d : c ) {
						
						int idx = samples.indexOf(d);
												
						gs[i++] = geoms.get( idx );
								
						for( int j = 0; j < fa.length; j++ )
								sfa[j] += oSamples.get(idx)[fa[j]];
												
						sum_fips += d[FIPS]+",";
					}
					
					for( double d : sfa )
						featureBuilder.add(d);
					featureBuilder.add(sum_fips);
					Geometry g = gf.createGeometryCollection(gs).union();
					featureBuilder.add(g);
					fc.add( featureBuilder.buildFeature(fc.size()+""));
					
					nSamples.add(sfa);
					nGeoms.add(g);
					
					oMap.put(sfa,c);
				}
				
				log.debug("build new contiguity map");
				Map<double[],Set<double[]>> ncm = new HashMap<double[],Set<double[]>>();
				for( double[] a : oMap.keySet() ) {
					ncm.put(a, new HashSet<double[]>() );
					
					for( double[] b : oMap.keySet() ) {
						
						for( double[] c : oMap.get(a) )
							for( double[] d : oMap.get(b) )
								if( cm.get(c).contains(d) )
									ncm.get(a).add(b);
					}
						
				}
				log.debug("done.");
								
				// store shape file 
				/*try {
					
					log.debug("Store shape file...");
					Map map = Collections.singletonMap("url", new File("output/redcap_cng_"+n+"_"+gns+".shp").toURI().toURL());
					FileDataStoreFactorySpi factory = new ShapefileDataStoreFactory();
					DataStore myData = factory.createNewDataStore(map);
					myData.createSchema(fc.getSchema());
					String name = fc.getSchema().getName().getLocalPart();
					FeatureStore<SimpleFeatureType, SimpleFeature> store = (FeatureStore<SimpleFeatureType, SimpleFeature>) myData.getFeatureSource(name);

					store.addFeatures(fc);
					log.debug("done.");
				} catch (MalformedURLException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}*/
				
				/*try {
					Drawer.geoDrawCluster( cluster.values(), samples, geoms, new FileOutputStream("output/preproc_cng_"+n+"_"+gns+"a.png"), true );
					Drawer.geoDrawCluster( subcluster, samples, geoms, new FileOutputStream("output/preproc_cng_"+n+"_"+gns+"b.png"), true );
				} catch (FileNotFoundException e1) {
					e1.printStackTrace();
				}*/
				
				int nfa[] = new int[]{0};
				
				for( int i : nfa)
					DataUtils.zScoreColumn(nSamples, i );
								
				log.debug("greedy regionalization.");
				int numRegions = 30;
				List<Set<double[]>> nCluster = new ArrayList<Set<double[]>>(); // new 
				for( int i = 0; i < numRegions; i++ ) 
					nCluster.add( new HashSet<double[]>() );
				
				List<double[]> toAdd = new ArrayList<double[]>(nSamples);
				while( !toAdd.isEmpty() ) {
										
					log.debug("todo: "+toAdd.size() );
					
					List<double[]> redo = new ArrayList<double[]>();
										
					for( double[] d : toAdd ) {
						Set<double[]> best = null;
						double bestHeterogenity = Double.MAX_VALUE;
						
						for( Set<double[]> s : nCluster ) {
							s.add(d);
							
							if( RegionUtils.isContiugous(ncm, s) && ( best == null || RegionUtils.getHeterogenity(nCluster, nfa) < bestHeterogenity ) ) {
								best = s;
								bestHeterogenity = RegionUtils.getHeterogenity(nCluster, nfa);
							}
							s.remove(d);
						}
																		
						if( best != null ) 
							best.add(d);
						else
							redo.add(d); 
					}
					toAdd = redo;
					
				}	
				log.debug("done.");
												
				/*try {
					Drawer.geoDrawCluster( nCluster, nSamples, nGeoms, new FileOutputStream("output/preproc_cng_"+n+"_"+gns+"c.png"), true );
				} catch (FileNotFoundException e1) {
					e1.printStackTrace();
				}*/
				
				log.debug("reconstruct clusters.");
				List<Set<double[]>> fCluster = new ArrayList<Set<double[]>>();
				for( Set<double[]> c : nCluster ) {
					Set<double[]> s = new HashSet<double[]>();
					for(double[] d : c )
						s.addAll( oMap.get(d) );
					fCluster.add(s);
				}
				log.debug("done.");
				
				log.debug("Heterogenity: "+RegionUtils.getHeterogenity(fCluster, fa));
				
				/*try {
					Drawer.geoDrawCluster( fCluster, samples, geoms, new FileOutputStream("output/preproc_cng_"+n+"_"+gns+"d.png"), true );
				} catch (FileNotFoundException e1) {
					e1.printStackTrace();
				}*/
			}		
		}
	}
}
