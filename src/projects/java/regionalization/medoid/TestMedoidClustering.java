package regionalization.medoid;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;

import heuristics.tabu.TabuSearch;
import regionalization.medoid.MedoidRegioClustering.GrowMode;
import regionalization.nga.WSSCutsTabuEvaluator;
import regionalization.nga.tabu.CutsTabuIndividual;
import spawnn.dist.ConstantDist;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GraphUtils;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class TestMedoidClustering {

	private static Logger log = Logger.getLogger(TestMedoidClustering.class);

	public enum method {
		hc, medoid
	};
	
	public enum treeCutMethod {
		tabu, redcap, cuts
	};
	
	public enum mInit {
		rnd, fDist, gDist, graphDist
	}

	public static void main(String[] args) {

		// redcap
		//SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("output/selection.shp"), true);
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/redcap/Election/election2004.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		//Map<double[], Set<double[]>> cm = GraphUtils.deriveQueenContiguitiyMap(samples, geoms,false); // some islands are not connected
		Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		for( Entry<double[],Set<double[]>> e : cm.entrySet() ) // no identity
			e.getValue().remove(e.getKey());
				
		for (int i = 0; i < samples.size(); i++) {
			Coordinate c = geoms.get(i).getCentroid().getCoordinate();
			samples.get(i)[0] = c.x;
			samples.get(i)[1] = c.y;
		}
		DataUtils.transform(samples, new int[]{7}, DataUtils.transform.zScore ); // not needed	
		Dist<double[]> fDist = new EuclideanDist(new int[] { 7 });
		Dist<double[]> gDist = new EuclideanDist(new int[] { 0,1 });
						
		// census
		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("output/counties.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		Map<double[], Set<double[]>> cm = GraphUtils.deriveQueenContiguitiyMap(sdf.samples, sdf.geoms,false);
		
		int[] fa = new int[] { 2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
		DataUtils.transform(samples, fa, DataUtils.transform.zScore );
		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(new int[] { 0, 1 });*/
		
		class Result {
			Object[][] params;
			Map<Object,Collection<Set<double[]>>> cluster = new HashMap<>();
			DescriptiveStatistics ds = null;
		}

		List<Object[][]> params = new ArrayList<Object[][]>();
		for( int numCluster : new int[]{ 7 } ) {
		//for( int numCluster = 2; numCluster <= 30; numCluster++ ) {
			// hc
			for( HierarchicalClusteringType type : new HierarchicalClusteringType[]{ /*HierarchicalClusteringType.single_linkage, HierarchicalClusteringType.complete_linkage, HierarchicalClusteringType.average_linkage/*, HierarchicalClusteringType.ward*/ } )
				params.add( new Object[][]{ 
					new Object[]{ numCluster }, // general
					new Object[]{ method.hc, type }, // tree-build-method
					new Object[]{ treeCutMethod.redcap, treeCutMethod.cuts, /*treeCutMethod.tabu*/ } // cluster-params
				} );
			
			// medoid		
			for( boolean gd : new boolean[]{ false, true } )
			for( GrowMode dm : new GrowMode[]{ /*DistMode.WSS,*/ GrowMode.EuclideanSqrd, /*DistMode.WSS_INC*/ } )
				for( mInit mi : new mInit[]{ mInit.rnd, mInit.fDist, mInit.gDist/*, mInit.graphDist*/ } )
				params.add( new Object[][]{
					new Object[]{ numCluster },
					new Object[]{ method.medoid, dm, 10, mi, gd }, 
					new Object[]{ "none" } 
				});
		}
		
		int threads = 4;
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<Result>> futures = new ArrayList<Future<Result>>();
		for (final Object[][] p : params )
			futures.add(es.submit(new Callable<Result>() {
				@Override
				public Result call() throws Exception {
					Result r = new Result();
					r.params = p;
					
					int numCluster = (Integer)p[0][0];
					
					method me = (method)p[1][0];
					if( me == method.hc ) {
						HierarchicalClusteringType type = (HierarchicalClusteringType) p[1][1];
						Map<Set<double[]>, TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, fDist, type );
						for( Object o : p[2] ) {
							treeCutMethod tcm = (treeCutMethod)o;
							if( tcm == treeCutMethod.cuts )
								r.cluster.put(o, Clustering.cutTree(tree, numCluster));
							else if( tcm == treeCutMethod.redcap ) {
								r.cluster.put(o, Clustering.cutTreeREDCAP(tree.values(), cm, type, numCluster, fDist) );
							} else if( tcm == treeCutMethod.tabu ){
								CutsTabuIndividual.k = -1;
								TabuSearch.rndMoveDiversication = true;
								WSSCutsTabuEvaluator evaluator = new WSSCutsTabuEvaluator(fDist);
								TabuSearch<CutsTabuIndividual> ts = new TabuSearch<>(evaluator, 10, 350, 25, 25 );
								CutsTabuIndividual init = new CutsTabuIndividual(Clustering.toREDCAPSpanningTree(tree.values(), cm, type, fDist), numCluster);
								r.cluster.put(o, ts.search(init).toCluster());
							} 							
						}
					} else if( me == method.medoid ) {
						Random ra = new Random();
						GrowMode dm = (GrowMode)p[1][1];
						mInit mi = (mInit)p[1][3];
						r.ds = new DescriptiveStatistics();
						double bestCost = Double.MAX_VALUE;
						for( int i = 0; i < (int)p[1][2]; i++ ) { // random restarts
							Set<double[]> medoids = new HashSet<double[]>();
							
							if( mi == mInit.rnd ) { // rnd init
								while( medoids.size() < numCluster ) {
									for (double[] s : cm.keySet())
										if (ra.nextDouble() < 1.0 / cm.keySet().size() ) {
											medoids.add(s);
											break;
										}
								}
							} else if( mi == mInit.fDist || mi == mInit.gDist || mi == mInit.graphDist ) { //k-means++-init
								medoids.add(samples.get(ra.nextInt(samples.size())));
								Map<double[],Map<double[],Double>> wCm = GraphUtils.toWeightedGraph(cm,new ConstantDist<>(1.0));
								
								Map<double[],Map<double[],Double>> map = new HashMap<>();
								
								while( medoids.size() < numCluster ) {
									// build dist map
									Map<double[],Double> distMap = new HashMap<>();
									for( double[] n : samples) {
										if( medoids.contains(n) )
											continue;
										
										double d = Double.MAX_VALUE;
										if( mi == mInit.graphDist ) {
											for( double[] m : medoids ) {
												if( !map.containsKey(m) )
													map.put(m, GraphUtils.getShortestDists(wCm, m));
												d = Math.min(d, map.get(m).get(n) );
											}
										} else {
											for( double[] m : medoids )
												if( mi == mInit.fDist )
													d = Math.min( d, fDist.dist(n, m));
												else if( mi == mInit.gDist )
													d = Math.min( d, gDist.dist(n, m));
										}
										distMap.put( n, d );
									}
											
									// tournament selection
									double min = Collections.min(distMap.values());
									double max = Collections.max(distMap.values());
									double sum = 0;
									for( double d : distMap.values() )
										sum += Math.pow((d - min)/(max-min),2); 
										
									double v = ra.nextDouble() * sum;
									double lower = 0;
									for (Entry<double[], Double> e : distMap.entrySet()) {
										double w = Math.pow((e.getValue()-min)/(max-min),2);
										if (lower <= v && v <= lower + w ) {
											medoids.add(e.getKey());
											break;
										}
										lower += w;
									}		
									
								}
							} 
									
							Map<double[], Set<double[]>> c;
							if( (boolean)p[1][4] )
								c = MedoidRegioClustering.cluster(cm, medoids, fDist, dm, 20, gDist );
							else 
								c = MedoidRegioClustering.cluster(cm, medoids, fDist, dm, 20, null );
							
							double cost = DataUtils.getWithinSumOfSquares(c.values(), fDist);
							r.ds.addValue(cost);
							if( cost < bestCost ) {
								r.cluster.put(p[2][0],c.values());
								bestCost = cost;
							}
						}
					} 
					return r;
				}
			}));

		es.shutdown();

		FileWriter fw = null;
		try {
			fw = new FileWriter("output/results.csv");
			for (Future<Result> ff : futures) {
				try {
					Result r = ff.get();
					for( Object o : r.params[2] ) {
						String s = Arrays.toString(r.params[0])+","+Arrays.toString(r.params[1])+","+o;
						s = s.replaceAll("[\\[\\] ]", "");
						if( r.ds != null )
							s += ", mean "+r.ds.getMean()+", min "+r.ds.getMin();
						log.info(s+","+DataUtils.getWithinSumOfSquares(r.cluster.get(o), fDist));
						fw.write(s+","+DataUtils.getWithinSumOfSquares(r.cluster.get(o), fDist)+"\n");
						Drawer.geoDrawCluster(r.cluster.get(o), samples, geoms, "output/"+s+".png", true);
					}
				} catch (InterruptedException ex) {
					ex.printStackTrace();
				} catch (ExecutionException ex) {
					ex.printStackTrace();
				}
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		} finally {
			try {
				fw.close();
			} catch (IOException e1) {
				e1.printStackTrace();
			}
		}
	}
}
