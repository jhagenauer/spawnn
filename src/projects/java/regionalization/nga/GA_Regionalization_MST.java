package regionalization.nga;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

import heuristics.Evaluator;
import heuristics.GeneticAlgorithm;
import heuristics.tabu.TabuSearch;
import regionalization.nga.tabu.CutsTabuIndividual;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.RandomDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.GraphUtils;

public class GA_Regionalization_MST {
	private static Logger log = Logger.getLogger(GA_Regionalization_MST.class);

	public static void main(String[] args) {
		GeometryFactory gf = new GeometryFactory();
		Random r = new Random();
		
		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/redcap/Election/election2004.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		int numCluster = 12;
		Dist<double[]> fDist = new EuclideanDist(new int[] { 7 });*/
		
		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();
		List<Coordinate> coords = new ArrayList<Coordinate>();
		while (samples.size() < 400) {
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
		}
		
		int numCluster = 5;
		Dist<double[]> fDist = new EuclideanDist(new int[] { 2 });
		
		Dist<double[]> rDist = new RandomDist<>();
		
		log.debug("start");
		List<TreeNode> hcTree = Clustering.getHierarchicalClusterTree(cm, fDist, HierarchicalClusteringType.ward);
		List<Set<double[]>> hcClusters = Clustering.treeToCluster( Clustering.cutTree(hcTree, numCluster) );
		double wardCost = DataUtils.getWithinSumOfSquares(hcClusters, fDist);
		log.debug("ward: " + wardCost );
				
		int repeats = 1;
		{
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( int i = 0; i < repeats; i++ ) {
				Map<double[], Set<double[]>> mst = GraphUtils.getMinimumSpanningTree(cm, fDist);
				List<Set<double[]>> skaterClusters = Clustering.skater(mst, numCluster - 1, fDist, 0);
				ds.addValue( DataUtils.getWithinSumOfSquares(skaterClusters, fDist));
			}
			log.debug("skater: "+ds.getMean()+","+ds.getMin());
		}
		
		{
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( int i = 0; i < repeats; i++ ) {
				List<TreeNode> m = Clustering.getHierarchicalClusterTree(cm, fDist, Clustering.HierarchicalClusteringType.ward);
				List<Set<double[]>> c = Clustering.cutTreeREDCAP(m, cm, HierarchicalClusteringType.ward, numCluster, fDist);
				ds.addValue( DataUtils.getWithinSumOfSquares(c, fDist));
			}
			log.debug("redcap(ward): "+ds.getMean()+","+ds.getMin());
		}
				
		{
			int lower = 0;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( int i = 0; i < 10; i++ ) {
				double cost = 0; //DataUtils.getWithinSumOfSquares(MedoidRegioClustering.cluster(cm, numCluster, fDist, DistMode.WSS, 20 ).values(), fDist);
				if( cost < wardCost ) lower++;
				ds.addValue( cost );
			}
			log.debug("pam WSS: "+ds.getMean()+","+ds.getMin()+","+(double)lower/repeats);
		}
				
		{
			int lower = 0;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( int i = 0; i < 10; i++ ) {
				double cost = 0; //DataUtils.getWithinSumOfSquares(MedoidRegioClustering.cluster(cm, numCluster, fDist, DistMode.EuclideanSqrd, 20 ).values(), fDist);
				if( cost < wardCost ) lower++;
				ds.addValue( cost );
			}
			log.debug("pam dist^2: "+ds.getMean()+","+ds.getMin()+","+(double)lower/repeats);
		}
		
		{
			TabuSearch.rndMoveDiversication = true;
			CutsTabuIndividual.k = -1;
			Evaluator<CutsTabuIndividual> evaluator = new WSSCutsTabuEvaluator(fDist);
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				CutsTabuIndividual init = new CutsTabuIndividual(GraphUtils.getMinimumSpanningTree(cm, fDist), numCluster );
				
				TabuSearch<CutsTabuIndividual> ts = new TabuSearch<CutsTabuIndividual>(evaluator,10, 350, 50, 25 );
				CutsTabuIndividual ti = ts.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best tabu_mst wPen -1: " + ds.getMean() + "," + ds.getMin());
		}
		
		{
			TabuSearch.rndMoveDiversication = true;
			CutsTabuIndividual.k = -1;
			Evaluator<CutsTabuIndividual> evaluator = new WSSCutsTabuEvaluator(fDist);
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				Collection<TreeNode> st = Clustering.getHierarchicalClusterTree(cm, fDist, HierarchicalClusteringType.ward);
				Map<double[],Set<double[]>> tree = Clustering.toREDCAPSpanningTree(st, cm, HierarchicalClusteringType.ward, fDist);
				CutsTabuIndividual init = new CutsTabuIndividual(tree, numCluster );
				
				TabuSearch<CutsTabuIndividual> ts = new TabuSearch<CutsTabuIndividual>(evaluator,10, 350, 50, 25 );
				CutsTabuIndividual ti = ts.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best tabu_ward wPen -1: " + ds.getMean() + "," + ds.getMin());
		}
		
		Evaluator<TreeIndividual> evaluator = new WSSEvaluator(fDist);
		GeneticAlgorithm.debug = true;
		{
			TreeIndividual.k = 0;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, fDist); //constant trees
					init.add(new TreeIndividual(tree, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best ga_mst 0: " + ", "+ ds.getMean() + "," + ds.getMin());
		}
		
		{
			TreeIndividual.k = -1;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, fDist); //constant trees
					init.add(new TreeIndividual(tree, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best ga_mst -1: " + ", "+ ds.getMean() + "," + ds.getMin());
		}
		
		{
			TreeIndividual.k = 0;
			TreeIndividual.mutCut = true;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, fDist); //constant trees
					init.add(new TreeIndividual(tree, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best ga_mst 0 mutcut: " + ", "+ ds.getMean() + "," + ds.getMin());
		}
		
		{
			TreeIndividual.k = -1;
			TreeIndividual.mutCut = true;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, fDist); //constant trees
					init.add(new TreeIndividual(tree, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best ga_mst -1 mutCut: " + ", "+ ds.getMean() + "," + ds.getMin());
		}
		
		{
			TreeIndividual.k = 0;
			TreeIndividual.mutCut = false;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, rDist); 
					init.add(new TreeIndividual(tree, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best ga_r 0: " + ", "+ ds.getMean() + "," + ds.getMin());
		}
		
		{
			TreeIndividual.k = -1;
			TreeIndividual.mutCut = false;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, rDist); 
					init.add(new TreeIndividual(tree, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best ga_r -1: " + ", "+ ds.getMean() + "," + ds.getMin());
		}
		
		{
			TreeIndividual.k = 0;
			TreeIndividual.mutCut = true;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, rDist); 
					init.add(new TreeIndividual(tree, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best ga_r 0 mutCut: " + ", "+ ds.getMean() + "," + ds.getMin());
		}
		
		{
			TreeIndividual.k = -1;
			TreeIndividual.mutCut = true;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, rDist); 
					init.add(new TreeIndividual(tree, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				ds.addValue(evaluator.evaluate(ti));
			}
			log.debug("best ga_r -1 mutCut: " + ", "+ ds.getMean() + "," + ds.getMin());
		}
	}
}
