package regionalization.nga;

import java.io.File;
import java.util.ArrayList;
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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;

import regionalization.medoid.MedoidRegioClustering;
import regionalization.medoid.MedoidRegioClustering.GrowMode;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.transform;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class GridSearchRegionalizationMedoid {

	private static Logger log = Logger.getLogger(GridSearchRegionalizationMedoid.class);

	public static void main(String[] args) {

		class Data {
			SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/redcap/Election/election2004.shp"), true);
			List<double[]> samples = sdf.samples;
			List<Geometry> geoms = sdf.geoms;
			Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
			
			Data() {
				for( int i = 0; i < samples.size(); i++ ) {
					Coordinate c = geoms.get(i).getCentroid().getCoordinate();
					samples.get(i)[0] = c.x;
					samples.get(i)[1] = c.y;
				}
			}

			/*List<double[]> samples = new ArrayList<double[]>();
			List<Geometry> geoms = new ArrayList<Geometry>();
			List<Coordinate> coords = new ArrayList<Coordinate>();
			List<Geometry> voroGeoms = new ArrayList<Geometry>();
			Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();

			public Data() {
				GeometryFactory gf = new GeometryFactory();
				Random r = new Random();
				samples = new ArrayList<double[]>();
				geoms = new ArrayList<Geometry>();
				coords = new ArrayList<Coordinate>();
				while (samples.size() < 500) {
					double x = r.nextDouble();
					double y = r.nextDouble();
					double z = r.nextDouble();
					Coordinate c = new Coordinate(x, y);
					coords.add(c);
					geoms.add(gf.createPoint(c));
					samples.add(new double[] { x, y, z });
				}

				VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
				vdb.setClipEnvelope(new Envelope(0, 0, 1, 1));
				vdb.setSites(coords);
				GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

				voroGeoms = new ArrayList<Geometry>();
				for (int i = 0; i < coords.size(); i++) {
					Geometry p = gf.createPoint(coords.get(i));
					for (int j = 0; j < coll.getNumGeometries(); j++)
						if (p.intersects(coll.getGeometryN(j))) {
							voroGeoms.add(coll.getGeometryN(j));
							break;
						}
				}

				// build cm map based on voro
				cm = new HashMap<double[], Set<double[]>>();
				for (int i = 0; i < samples.size(); i++) {
					Set<double[]> s = new HashSet<double[]>();
					for (int j = 0; j < samples.size(); j++)
						if (i != j && voroGeoms.get(i).intersects(voroGeoms.get(j)))
							s.add(samples.get(j));
					cm.put(samples.get(i), s);
				}
			}*/
		}

		/*int numCluster = 5;
		Dist<double[]> fDist = new EuclideanDist(new int[] { 2 });
		Dist<double[]> gDist = new EuclideanDist(new int[] { 0,1 });*/
				
		int numCluster = 12;
		Dist<double[]> fDist = new EuclideanDist(new int[] { 7 });
		Dist<double[]> gDist = new EuclideanDist(new int[] { 0,1 });

		int rndRestarts = 1;
		int threads = 4;
		int runs = 4;
		List<Data> data = new ArrayList<Data>();
		while (data.size() < 1)
			data.add(new Data());

		double slk = 0;
		for( Data dt : data ) {
			List<TreeNode> hcTree = Clustering.getHierarchicalClusterTree(dt.cm, fDist, HierarchicalClusteringType.single_linkage);
			List<Set<double[]>> hcClusters = Clustering.cutTree(hcTree, numCluster);
			slk += DataUtils.getWithinSumOfSquares(hcClusters, fDist);
		}
		log.debug("slk: "+slk/data.size());
		
		double avg = 0;
		for( Data dt : data ) {
			List<TreeNode> hcTree = Clustering.getHierarchicalClusterTree(dt.cm, fDist, HierarchicalClusteringType.average_linkage);
			List<Set<double[]>> hcClusters = Clustering.cutTree(hcTree, numCluster);
			avg += DataUtils.getWithinSumOfSquares(hcClusters, fDist);
		}
		log.debug("avg: "+avg/data.size());
		
		double clk = 0;
		for( Data dt : data ) {
			List<TreeNode> hcTree = Clustering.getHierarchicalClusterTree(dt.cm, fDist, HierarchicalClusteringType.complete_linkage);
			List<Set<double[]>> hcClusters = Clustering.cutTree(hcTree, numCluster);
			clk += DataUtils.getWithinSumOfSquares(hcClusters, fDist);
		}
		log.debug("clk: "+clk/data.size());
		
		double ward = 0;
		for( Data dt : data ) {
			List<TreeNode> hcTree = Clustering.getHierarchicalClusterTree(dt.cm, fDist, HierarchicalClusteringType.ward);
			List<Set<double[]>> hcClusters = Clustering.cutTree(hcTree, numCluster);
			ward += DataUtils.getWithinSumOfSquares(hcClusters, fDist);
		}
		log.debug("ward: "+ward/data.size());
		
		System.exit(1);

		for( final GrowMode dm : new GrowMode[]{ GrowMode.WSS/*, DistMode.Euclidean, DistMode.EuclideanSqrt*/ } )
			for( final int initMode : new int[]{ 0, 1, 2 } ) {
					long time = System.currentTimeMillis();

					ExecutorService es = Executors.newFixedThreadPool(threads);
					List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

					for (final Data dt : data) {
						for (int i = 0; i < runs; i++) {
							futures.add(es.submit(new Callable<double[]>() {
								@Override
								public double[] call() throws Exception {
									double cost = Double.MAX_VALUE;
									
									for( int i = 0; i < rndRestarts; i++ ) { // random restarts
										Random r = new Random();
										Set<double[]> init;
										if( initMode == 0 ) { // random init
											init = new HashSet<double[]>();
											while( init.size() < numCluster ) {
												for (double[] s : dt.samples)
													if (r.nextDouble() < 1.0 / dt.samples.size() ) {
														init.add(s);
														break;
													}
											}
										} else if( initMode == 1 ) { // medoid g-dist
											init = Clustering.kMedoidsPAM(dt.samples, numCluster, gDist).keySet();
										} else { // medoid f-dist
											init = Clustering.kMedoidsPAM(dt.samples, numCluster, fDist).keySet();
										}
										cost = Math.min(cost, DataUtils.getWithinSumOfSquares(MedoidRegioClustering.cluster(dt.cm, init, fDist, GrowMode.WSS, 20 ).values(), fDist));
									}
									
									return new double[]{cost}; 
								}
							}));
						}
					}
					es.shutdown();

					DescriptiveStatistics ds = new DescriptiveStatistics();
					for (Future<double[]> ff : futures) {
						try {
							ds.addValue(ff.get()[0]);
						} catch (InterruptedException ex) {
							ex.printStackTrace();
						} catch (ExecutionException ex) {
							ex.printStackTrace();
						}
					}
					log.info(dm + "\t"+initMode+"\t mean: " + ds.getMean() + "\t min: " + ds.getMin() +"\t st dev:" + ds.getStandardDeviation() + "\t" + (System.currentTimeMillis() - time) / 1000.0);
				}
	}
}
