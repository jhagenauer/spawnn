package regionalization.nga;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

import heuristics.Evaluator;
import heuristics.GeneticAlgorithm;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.RandomDist;
import spawnn.utils.Drawer;
import spawnn.utils.GraphUtils;

public class GA_Regionalization_Test {
	private static Logger log = Logger.getLogger(GA_Regionalization_Test.class);

	public static void main(String[] args) {
		GeometryFactory gf = new GeometryFactory();
		Random r = new Random();

		int k = 0;
		while (true) {
			log.debug(k++);

			List<double[]> samples = new ArrayList<double[]>();
			List<Geometry> geoms = new ArrayList<Geometry>();
			List<Coordinate> coords = new ArrayList<Coordinate>();
			while (samples.size() < 4) {
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

			int numCluster = 2;
			Dist<double[]> fDist = new EuclideanDist(new int[] { 2 });
			Dist<double[]> rDist = new RandomDist<>();

			Evaluator<TreeIndividual> evaluator = new WSSEvaluator(fDist);
			GeneticAlgorithm.debug = false;

			int repeats = 10;
			TreeIndividual bestA = null, bestB = null;

			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, fDist); // constant trees
					init.add(new TreeIndividual(cm, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				if (bestA == null || ti.getValue() < bestA.getValue())
					bestA = ti;
			}

			for (int i = 0; i < repeats; i++) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while (init.size() < 50) {
					Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(cm, rDist);
					init.add(new TreeIndividual(cm, tree, numCluster));
				}
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual ti = ga.search(init);
				if (bestB == null || ti.getValue() < bestB.getValue())
					bestB = ti;
			}
			
			if( bestB.getValue()+0.0001 < bestA.getValue() ) {
				log.debug(samples.size()+", bestA: "+bestA.getValue()+", bestB: "+bestB.getValue() );
				
				Drawer.geoDrawConnections(bestA.getTree(), bestA.getCuts(), new int[]{0,1}, null, "output/bestA_cuts.png");
				Drawer.geoDrawWeightedConnections( GraphUtils.getWeightedGraph(bestA.getTree(),fDist), new int[]{0,1}, null, "output/bestA.png");
				
				Drawer.geoDrawConnections(bestB.getTree(), bestB.getCuts(), new int[]{0,1}, null, "output/bestB_cuts.png");
				Drawer.geoDrawWeightedConnections( GraphUtils.getWeightedGraph(bestB.getTree(),fDist), new int[]{0,1}, null, "output/bestB.png");
				
				Drawer.geoDrawConnections(cm, null, new int[]{0,1}, null, "output/cm.png");
				break;
			}
		}
	}
}
