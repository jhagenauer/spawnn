package regionalization.nga;

import java.util.ArrayList;
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

import myga.Evaluator;
import myga.GeneticAlgorithm;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.RandomDist;
import spawnn.utils.Clustering;
import spawnn.utils.Drawer;

public class GA_MST {
	private static Logger log = Logger.getLogger(GA_MST.class);

	public static void main(String[] args) {
		GeometryFactory gf = new GeometryFactory();
		Random r = new Random();
		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();
		List<Coordinate> coords = new ArrayList<Coordinate>();
		while( samples.size() < 80 ) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			double z = r.nextDouble();
			Coordinate c = new Coordinate(x, y);
			coords.add(c);
			geoms.add( gf.createPoint(c));
			samples.add( new double[]{ x,y,z } );
		}
		
		VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
		vdb.setClipEnvelope(new Envelope(0, 0, 1, 1) );
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
		Drawer.geoDrawConnections(cm, null, new int[]{0,1}, null, "output/cm.png");
		
		Dist<double[]> fDist = new EuclideanDist(new int[]{2});
		Dist<double[]> rDist = new RandomDist<double[]>();
		
		Evaluator<TreeIndividual> evaluator = new MSTEvaluator(fDist);
		TreeIndividual mst = new TreeIndividual(cm, Clustering.getMinimumSpanningTree(cm, fDist));
		Drawer.geoDrawConnections(mst.getTree(), null, new int[]{0,1}, null, "output/mst.png");
		log.debug("mst: "+evaluator.evaluate(mst));
		
		{
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( int j = 0; j < 1; j++ ) {
				TreeIndividual bestRnd = null;
				for( int i = 0; i < 10000; i++ ) {
					TreeIndividual ti = new TreeIndividual(cm, Clustering.getMinimumSpanningTree(cm, rDist));
					if( bestRnd == null || evaluator.evaluate(ti) < evaluator.evaluate(bestRnd))
						bestRnd = ti;
				}
				ds.addValue(evaluator.evaluate(bestRnd));
			}
			log.debug("best rnd: "+ds.getMean()+","+ds.getMin());
		}
		
		{
			GeneticAlgorithm.debug = true;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( int j = 0; j < 1; j++ ) {
				List<TreeIndividual> init = new ArrayList<TreeIndividual>();
				while( init.size() < 50 ) 
					init.add( new TreeIndividual(cm, Clustering.getMinimumSpanningTree(cm, rDist)));
				GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
				TreeIndividual bestGA = ga.search(init);
				ds.addValue(evaluator.evaluate(bestGA));
			}
			log.debug("best ga: "+ds.getMean()+","+ds.getMin());
		}
	}
}
