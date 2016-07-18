package regionalization;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Map.Entry;

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
import regionalization.medoid.MedoidRegioClustering;
import regionalization.medoid.MedoidRegioClustering.GrowMode;
import regionalization.medoid.ga.MedoidIndividual;
import regionalization.medoid.ga.WSSEvaluator;
import regionalization.nga.TreeIndividual;
import regionalization.nga.tabu.CutsTabuIndividual;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.RandomDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.GraphUtils;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class TestRegionalizationAlgorithms {
	private static Logger log = Logger.getLogger(TestRegionalizationAlgorithms.class);

	public static void main(String[] args) {
		GeometryFactory gf = new GeometryFactory();
		Random r = new Random();
		
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
		DataUtils.transform(samples, new int[]{7}, DataUtils.Transform.zScore ); // not needed	
		Dist<double[]> fDist = new EuclideanDist(new int[] { 7 });
		Dist<double[]> gDist = new EuclideanDist(new int[] { 0,1 });
		Dist<double[]> rDist = new RandomDist<>();
		
		
		int numCluster = 7;
		
		{
			List<TreeNode> hcTree = Clustering.getHierarchicalClusterTree(cm, fDist, HierarchicalClusteringType.ward);
			long time = System.currentTimeMillis();
			List<Set<double[]>> cutsCluster = Clustering.cutTree(hcTree, numCluster);
			log.debug("took: "+(double)(System.currentTimeMillis()-time)/1000.0);
			log.debug("wardCuts: " + DataUtils.getWithinSumOfSquares(cutsCluster, fDist) );
			
			time = System.currentTimeMillis();
			Map<double[],Set<double[]>> spTree = Clustering.toREDCAPSpanningTree(hcTree, cm, Clustering.HierarchicalClusteringType.ward, fDist);
			List<Set<double[]>> redcapCluster = Clustering.cutTreeREDCAP( spTree, numCluster, fDist);
			log.debug("took: "+(double)(System.currentTimeMillis()-time)/1000.0);
			log.debug("wardRedcap: "+DataUtils.getWithinSumOfSquares(redcapCluster, fDist));
		}
			
		{
			GeneticAlgorithm.debug = true;
			WSSEvaluator eval = new WSSEvaluator(samples, cm, fDist, GrowMode.EuclideanSqrd);
			List<MedoidIndividual> init = new ArrayList<MedoidIndividual>();
			while (init.size() < 20) 
				init.add(new MedoidIndividual( numCluster, samples.size() ) );
				
			GeneticAlgorithm<MedoidIndividual> ga = new GeneticAlgorithm<>(eval);
			MedoidIndividual ti = ga.search(init);
			
			log.debug("ga: "+ti.getValue());
		}
		System.exit(1);
		
		{
			GeneticAlgorithm.debug = true;
			TreeIndividual.k = -1;
			TreeIndividual.mutCut = false;
			List<TreeIndividual> init = new ArrayList<TreeIndividual>();
			while (init.size() < 20) {
				Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, rDist); 
				init.add(new TreeIndividual(tree, tree, numCluster));
			}
			Evaluator<TreeIndividual> evaluator = new regionalization.nga.WSSEvaluator(fDist,true);
			GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
			TreeIndividual ti = ga.search(init);
			log.debug("best: " + ", "+ti.getValue());
		}
		System.exit(1);
		
	}
}
