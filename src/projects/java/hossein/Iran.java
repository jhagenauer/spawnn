package hossein;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.FeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ClusterValidation;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.GraphClustering;
import spawnn.utils.SpatialDataFrame;

public class Iran {

	private static Logger log = Logger.getLogger(Iran.class);
	
	public static void main(String[] args) {
		final Random r = new Random();
		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/landslides1.csv"), new int[]{0,1}, new int[]{}, true);
				
		final int[] fa = new int[] { 2,3,4,5,6,7,8,9,10,11,12,13 };
		final int[] ga = new int[] { 0, 1 };

		final Dist<double[]> fDist = new EuclideanDist(fa);
		final Dist<double[]> gDist = new EuclideanDist(ga);

		DataUtils.transform(sdf.samples, fa, Transform.zScore);
		DataUtils.zScoreGeoColumns(sdf.samples, ga, gDist);
		
		final int t_max = 100000;
		
		ExecutorService es = Executors.newFixedThreadPool(3);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
				
		for( int run = 0; run < 10; run++ )
		for (int n : new int[]{48} )
			for (int l : new int[]{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,28,32,36,40,48 }) {
				
				final int[] pars = new int[]{run,n,l};

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						Sorter<double[]> secSorter = new DefaultSorter<>(fDist);
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, pars[2]);

						DecayFunction nbRate = new PowerDecay(pars[1] * 2.0 / 3.0, 0.01);
						DecayFunction lrRate1 = new PowerDecay(0.6, 0.005);

						List<double[]> neurons = new ArrayList<double[]>();
						while (neurons.size() < pars[1]) {
							double[] d = sdf.samples.get(r.nextInt(sdf.samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						NG ng = new NG(neurons, nbRate, lrRate1, sorter);
						for (int t = 0; t < t_max; t++) {
							double[] d = sdf.samples.get(r.nextInt(sdf.samples.size()));
							ng.train((double) t / t_max, d);
						}

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(sdf.samples, neurons, sorter);
						double qe = DataUtils.getMeanQuantizationError(bmus, fDist);
						double sqe = DataUtils.getMeanQuantizationError(bmus, gDist);
						
						Map<double[],Map<double[],Double>> graph = new HashMap<>();
						for( double[] d : sdf.samples ) {
							sorter.sort(d, neurons);					
							double[] n0 = neurons.get(0), n1 = neurons.get(1);
							
							if( !graph.containsKey( n0 ) )
								graph.put(n0, new HashMap<double[],Double>() );
							if( !graph.get(n0).containsKey(n1) )
								graph.get(n0).put(n1, 1.0);
							else
								graph.get(n0).put(n1, graph.get(n0).get(n1)+1.0);
							
							if( !graph.containsKey( n1) )
								graph.put(n1, new HashMap<double[],Double>() );
							if( !graph.get(n1).containsKey(n0) )
								graph.get(n1).put(n0, 1.0);
							else
								graph.get(n1).put(n0, graph.get(n1).get(n0)+1.0);
						}
						
						double max = 0;
						for( Map<double[],Double> m : graph.values() )
							max = Math.max( max, Collections.max(m.values() ) );
						
						Map<double[],Map<double[],Double>> nGraph = new HashMap<>();
						for( double[] n0 : graph.keySet() ) {
							Map<double[],Double> m = new HashMap<>();
							for( double[] n1 : graph.get(n0).keySet() )
								m.put(n1, graph.get(n0).get(n1)/max );
							nGraph.put(n0, m);
						}
												
						Map<double[],Integer> map = GraphClustering.multilevelOptimization(nGraph, 100 );
						
						Map<double[],Set<double[]>> ptCluster = new HashMap<>();
						for( Set<double[]> c :GraphClustering.modulMapToCluster(map) ) 
							ptCluster.put(DataUtils.getMean(c), c);
						
						double sil = ClusterValidation.getSilhouetteCoefficient(ptCluster, fDist);
						double mod = GraphClustering.modularity(nGraph, map);
												
						List<Set<double[]>> cluster = new ArrayList<>();
						for( Set<double[]> c : ptCluster.values() ) {
							Set<double[]> s = new HashSet<>();
							for( double[] pt : c )
								s.addAll(bmus.get(pt));
							cluster.add(s);
						}
												
						List<double[]> s = new ArrayList<>();
						for( double[] d : sdf.samples ) {
							double[] nd = Arrays.copyOf(d, d.length+1);
							for( int i = 0; i < cluster.size(); i++ )
								if( cluster.get(i).contains(d))
									nd[nd.length-1] = i+1;
							s.add(nd);
						}
						
						List<String> nn = new ArrayList<>(sdf.names);
						nn.add("cluster");
								
						DataUtils.writeShape(s, sdf.geoms, nn.toArray(new String[]{}), sdf.crs, "output/landslides1_"+pars[0]+"_"+pars[1]+"_"+pars[2]+".shp");
						Drawer.geoDrawCluster(cluster, sdf.samples, sdf.geoms, "output/landslides1_"+pars[0]+"_"+pars[1]+"_"+pars[2]+".png", true );
						
						return new double[]{pars[0],pars[1],pars[2],qe,sqe,sil,mod,ptCluster.size()};
					}
				}));
			}
		es.shutdown();

		for (Future<double[]> ff : futures) {
			try {
				double[] e = ff.get();
				log.info("run: "+e[0]+", neurons: "+e[1]+", l: "+e[2]+", quant error: "+e[3]+", spatial quant error: "+e[4]+", silhoutte: "+e[5]+", modularity: "+e[6]+", cluster size: "+e[7]);
			} catch (InterruptedException ex) {
				ex.printStackTrace();
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} 
		}		
	}
	
	public static <T> FeatureCollection<SimpleFeatureType, SimpleFeature> buildClusterFeatures(SpatialDataFrame sd, List<double[]> samples, List<Set<double[]>> cluster, int[] fa ) {
		SimpleFeatureTypeBuilder sftb = new SimpleFeatureTypeBuilder();
		sftb.setName("data");
		sftb.setCRS(sd.crs);
		for (int i : fa ) {
			if (sd.bindings.get(i) == SpatialDataFrame.binding.Double)
				sftb.add(sd.names.get(i), Double.class);
			else if (sd.bindings.get(i) == SpatialDataFrame.binding.Integer)
				sftb.add(sd.names.get(i), Integer.class);
			else if (sd.bindings.get(i) == SpatialDataFrame.binding.Long)
				sftb.add(sd.names.get(i), Long.class);
		}
		sftb.add("cluster", Integer.class);

		Geometry g = sd.geoms.get(0);
		if (g instanceof Polygon)
			sftb.add("the_geom", Polygon.class);
		else if (g instanceof MultiPolygon)
			sftb.add("the_geom", MultiPolygon.class);
		else if (g instanceof Point)
			sftb.add("the_geom", Point.class);
		else if (g instanceof MultiPoint)
			sftb.add("the_geom", MultiPoint.class);
		else
			throw new RuntimeException("Unkown geometry type!");

		SimpleFeatureType type = sftb.buildFeatureType();
		SimpleFeatureBuilder builder = new SimpleFeatureBuilder(type);

		DefaultFeatureCollection fc = new DefaultFeatureCollection();
		for( int i = 0; i < samples.size(); i++ ) {
			for( int k = 0; k < cluster.size(); k++ ) {
				if( cluster.get(k).contains(samples.get(i) ) ) {
					for (int j : fa )
						builder.set(sd.names.get(j), sd.samples.get(i)[j]);
					builder.set("cluster", k);
					builder.set("the_geom", sd.geoms.get(i));
					fc.add(builder.buildFeature(fc.size() + ""));
					break;
				}
			}
		}		
		return fc;
	}
}
