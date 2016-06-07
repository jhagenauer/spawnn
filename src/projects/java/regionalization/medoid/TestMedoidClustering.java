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
import regionalization.medoid.MedoidRegioClustering.MedoidInitMode;
import regionalization.nga.WSSCutsTabuEvaluator;
import regionalization.nga.tabu.CutsTabuIndividual;
import spawnn.dist.ConstantDist;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.RandomDist;
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
		hc, medoid, medoid2, medoid3
	};
	
	public enum treeCutMethod {
		tabu, redcap, cuts
	};
	
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
		Dist<double[]> rDist = new RandomDist<double[]>();
						
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
			
			for( GrowMode dm : new GrowMode[]{ GrowMode.EuclideanSqrd } )
			for( MedoidInitMode initMode : new MedoidInitMode[]{ MedoidInitMode.rnd } )
				params.add( new Object[][]{
					new Object[]{ numCluster },
					new Object[]{ method.medoid3, dm, 20, 1, initMode }, 
					new Object[]{ "none" } 
				});
			
			//medoid2
			for( GrowMode dm : new GrowMode[]{ } )
				for( MedoidInitMode initMode : new MedoidInitMode[]{ MedoidInitMode.rnd, MedoidInitMode.fDist, MedoidInitMode.gDist/*, mInit.graphDist*/ } )
				params.add( new Object[][]{
					new Object[]{ numCluster },
					new Object[]{ method.medoid2, dm, 10, initMode }, 
					new Object[]{ "none" } 
				});
			
			// medoid	
			for( boolean nbSearch : new boolean[]{ } )
			for( int maxNoImpro : new int[]{ 20 } )
			for( Dist updateDist : new Dist[]{ gDist, fDist, rDist } )
			for( GrowMode dm : new GrowMode[]{ /*DistMode.WSS,*/ GrowMode.EuclideanSqrd, /*GrowMode.WSS_INC*/ } )
				for( MedoidInitMode initMode : new MedoidInitMode[]{ MedoidInitMode.rnd, MedoidInitMode.fDist, MedoidInitMode.gDist/*, mInit.graphDist*/ } )
				params.add( new Object[][]{
					new Object[]{ numCluster },
					new Object[]{ method.medoid, dm, maxNoImpro, 30, initMode, updateDist, nbSearch }, 
					new Object[]{ "none" } 
				});
		}
		
		int threads = 10;
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
						GrowMode growMode = (GrowMode)p[1][1];
						int maxNoImpro = (int)p[1][2];						
						r.ds = new DescriptiveStatistics();
						double bestCost = Double.MAX_VALUE;
						for( int i = 0; i < (int)p[1][3]; i++ ) { // random restarts
							
							Set<double[]> medoids = MedoidRegioClustering.getInitMedoids((MedoidInitMode)p[1][4], cm, fDist, gDist, numCluster);
							Map<double[], Set<double[]>> c = MedoidRegioClustering.cluster(cm, medoids, fDist, growMode, maxNoImpro, (Dist<double[]>) p[1][5], (boolean)p[1][6] );
							
							double cost = DataUtils.getWithinSumOfSquares(c.values(), fDist);
							r.ds.addValue(cost);
							if( cost < bestCost ) {
								r.cluster.put(p[2][0],c.values());
								bestCost = cost;
							}
						}
					} else if( me == method.medoid2 ) {
						GrowMode growMode = (GrowMode)p[1][1];						
						r.ds = new DescriptiveStatistics();
						double bestCost = Double.MAX_VALUE;
						for( int i = 0; i < (int)p[1][2]; i++ ) { // random restarts
							Set<double[]> medoids = MedoidRegioClustering.getInitMedoids( (MedoidInitMode)p[1][3], cm, fDist, gDist, numCluster);
							Map<double[], Set<double[]>> c = MedoidRegioClustering.cluster2(cm, medoids, fDist, growMode );
							double cost = DataUtils.getWithinSumOfSquares(c.values(), fDist);
							r.ds.addValue(cost);
							if( cost < bestCost ) {
								r.cluster.put(p[2][0],c.values());
								bestCost = cost;
							}
						}
					} else if( me == method.medoid3 ) {
						GrowMode growMode = (GrowMode)p[1][1];						
						r.ds = new DescriptiveStatistics();
						double bestCost = Double.MAX_VALUE;
						for( int i = 0; i < (int)p[1][3]; i++ ) { // random restarts
							Set<double[]> medoids = MedoidRegioClustering.getInitMedoids( (MedoidInitMode)p[1][4], cm, fDist, gDist, numCluster);
							Map<double[], Set<double[]>> c = MedoidRegioClustering.cluster3(cm, medoids, fDist, growMode, (int)p[1][2] );
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
