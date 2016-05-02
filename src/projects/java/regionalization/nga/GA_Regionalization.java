package regionalization.nga;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

import myga.Evaluator;
import myga.GeneticAlgorithm;
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

public class GA_Regionalization {
	private static Logger log = Logger.getLogger(GA_Regionalization.class);

	public static void main(String[] args) {
		GeometryFactory gf = new GeometryFactory();
		Random r = new Random();
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/redcap/Election/election2004.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		
		/*List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();
		List<Coordinate> coords = new ArrayList<Coordinate>();
		while (samples.size() < 6) {
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

		List<Geometry> voroGeoms = new ArrayList<Geometry>();
		for (int i = 0; i < coords.size(); i++) {
			Geometry p = gf.createPoint(coords.get(i));
			for (int j = 0; j < coll.getNumGeometries(); j++)
				if (p.intersects(coll.getGeometryN(j))) {
					voroGeoms.add(coll.getGeometryN(j));
					break;
				}
		}

		// build cm map based on voro
		Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
		for (int i = 0; i < samples.size(); i++) {
			Set<double[]> s = new HashSet<double[]>();
			for (int j = 0; j < samples.size(); j++)
				if (i != j && voroGeoms.get(i).intersects(voroGeoms.get(j)))
					s.add(samples.get(j));
			cm.put(samples.get(i), s);
		}*/
		
		int numCluster = 12;
		Dist<double[]> fDist = new EuclideanDist(new int[] { 7 });
		Dist<double[]> rDist = new RandomDist<double[]>();

		Map<Set<double[]>, TreeNode> hcTree = Clustering.getHierarchicalClusterTree(cm, fDist, HierarchicalClusteringType.ward);
		List<Set<double[]>> hcClusters = Clustering.cutTree(hcTree, numCluster);
		log.debug("ward: " + DataUtils.getWithinSumOfSuqares(hcClusters, fDist));
		Drawer.geoDrawCluster(hcClusters, samples, geoms, "output/ward.png", true);
		
		{
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( int i = 0; i < 1; i++ ) {
				Map<double[], Set<double[]>> mst = GraphUtils.getMinimumSpanningTree(cm, fDist);
				List<Set<double[]>> skaterClusters = Clustering.skater(mst, numCluster - 1, fDist, 0);
				ds.addValue( DataUtils.getWithinSumOfSuqares(skaterClusters, fDist));
			}
			log.debug("skater: "+ds.getMean()+","+ds.getMin());
		}


		int repeats = 1;
		Evaluator<TreeIndividual> evaluator = new WSSEvaluator(fDist);
		GeneticAlgorithm.debug = true;
		
		TreeIndividual.onlyMutCuts = true;
		TreeIndividual.onlyMutTrees = false;
		{
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, fDist); //constant trees
					init.add(new TreeIndividual2(cm, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best ga_mst: " + ds.getMean() + "," + ds.getMin());
		}
				
		TreeIndividual.onlyMutCuts = false;
		TreeIndividual.onlyMutTrees = false;
		{
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, rDist); //constant trees
					init.add(new TreeIndividual2(cm, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best ga_norm: " + ds.getMean() + "," + ds.getMin());
		}
	}
}
