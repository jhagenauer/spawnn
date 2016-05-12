package regionalization.medoid;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;

import heuristics.tabu.TabuSearch;
import regionalization.medoid.MedoidRegioClustering.DistMode;
import regionalization.nga.WSSCutsTabuEvaluator;
import regionalization.nga.tabu.CutsTabuIndividual;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.transform;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class TestRegionalization {

	private static Logger log = Logger.getLogger(TestRegionalization.class);

	public enum method {
		hc, medoid
	};
	
	public enum treeCutMethod {
		tabu, redcap, cuts
	};

	public static void main(String[] args) {

		// redcap
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
		DataUtils.transform(samples, new int[]{7}, transform.zScore ); // not needed	
		Dist<double[]> fDist = new EuclideanDist(new int[] { 7 });
		Dist<double[]> gDist = new EuclideanDist(new int[] { 0,1 });
						
		// census
		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("output/counties.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		Map<double[], Set<double[]>> cm = GraphUtils.deriveQueenContiguitiyMap(sdf.samples, sdf.geoms);
		
		Dist<double[]> fDist = new EuclideanDist(new int[] { 2,3,4,5,6,7,8,9,10,11,12,13,14,15 });
		Dist<double[]> gDist = new EuclideanDist(new int[] { 0, 1 });*/
		
		class Result {
			Object[][] params;
			Map<Object,Collection<Set<double[]>>> cluster = new HashMap<>();
		}

		List<Object[][]> params = new ArrayList<Object[][]>();
		//for( int numCluster : new int[]{ 7 } ) {
		for( int numCluster = 2; numCluster <= 30; numCluster++ ) {
			// hc
			for( HierarchicalClusteringType type : new HierarchicalClusteringType[]{ HierarchicalClusteringType.single_linkage, HierarchicalClusteringType.complete_linkage, HierarchicalClusteringType.average_linkage, HierarchicalClusteringType.ward } )
				params.add( new Object[][]{ 
					new Object[]{ numCluster }, // general
					new Object[]{ method.hc, type }, // tree-build-method
					new Object[]{ treeCutMethod.redcap, treeCutMethod.cuts, treeCutMethod.tabu } // cluster-params
				} );
			
			// medoid
			for( DistMode dm : new DistMode[]{ DistMode.WSS, DistMode.EuclideanSqrd } )
				params.add( new Object[][]{
					new Object[]{ numCluster },
					new Object[]{ method.medoid, 10 }, // cluster-mode
					new Object[]{ dm } // dist-mode
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
						DistMode dm = (DistMode)p[2][0];
						double bestCost = Double.MAX_VALUE;
						for( int i = 0; i < (int)p[1][1]; i++ ) { // random restarts
							Map<double[], Set<double[]>> c = MedoidRegioClustering.clusterCached(cm, numCluster, fDist, dm );
							double cost = DataUtils.getWithinSumOfSquares(c.values(), fDist);
							if( cost < bestCost ) {
								r.cluster.put(dm,c.values());
								bestCost = cost;
							}
						}
					} 
					return r;
				}
			}));

		es.shutdown();

		for (Future<Result> ff : futures) {
			try {
				Result r = ff.get();
				for( Object o : r.params[2] ) {
					String s = r.params[0][0]+","+r.params[1][0]+","+o;
					s = s.replaceAll("[\\[\\] ]", "");
					log.info(s+","+DataUtils.getWithinSumOfSquares(r.cluster.get(o), fDist));
					Drawer.geoDrawCluster(r.cluster.get(o), samples, geoms, "output/"+s+".png", true);
				}
			} catch (InterruptedException ex) {
				ex.printStackTrace();
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			}
		}
	}
}
