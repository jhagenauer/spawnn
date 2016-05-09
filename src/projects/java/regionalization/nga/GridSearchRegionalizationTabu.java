package regionalization.nga;

import java.util.ArrayList;
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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

import heuristics.Evaluator;
import heuristics.tabu.TabuSearch;
import regionalization.nga.tabu.CutsTabuIndividual;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.GraphUtils;

public class GridSearchRegionalizationTabu {

	private static Logger log = Logger.getLogger(GridSearchRegionalizationTabu.class);

	public static void main(String[] args) {

		class Data {
			List<double[]> samples = new ArrayList<double[]>();
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
				while (samples.size() < 160) {
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
			}
		}

		Dist<double[]> fDist = new EuclideanDist(new int[] { 2 });
		int numCluster = 5;

		int threads = 4;
		int runs = 8;
		List<Data> data = new ArrayList<Data>();
		while (data.size() < 4)
			data.add(new Data());
		final Evaluator<CutsTabuIndividual> evaluator = new WSSCutsTabuEvaluator(fDist);

		double bestValue = Double.MAX_VALUE;
		
		for (final int k : new int[] { 0 })
			for (final int tlLength : new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24, 25, 26,27,28,29,30 })
				for (final int penDur : new int[] { 10, 25, 50, 75 }) 
					for( final int noImproUntilPen : new int[]{ 10, 25, 50, 75, 100 } ){
					long time = System.currentTimeMillis();

					CutsTabuIndividual.k = k;

					ExecutorService es = Executors.newFixedThreadPool(threads);
					List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

					for (final Data dt : data) {
						Map<double[], Set<double[]>> tree = GraphUtils.getMinimumSpanningTree(dt.cm, fDist);
						for (int i = 0; i < runs; i++) {
							futures.add(es.submit(new Callable<double[]>() {
								@Override
								public double[] call() throws Exception {
									CutsTabuIndividual init = new CutsTabuIndividual(tree, numCluster);
									TabuSearch<CutsTabuIndividual> ts = new TabuSearch<CutsTabuIndividual>(evaluator,tlLength, 350, noImproUntilPen, penDur );
									CutsTabuIndividual ti = ts.search(init);
									return new double[] { ti.getValue() };
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
					log.info(k + "\t" + tlLength + "\t" + noImproUntilPen +"\t" + penDur + "\t" + ds.getMean() + "\t" + ds.getStandardDeviation() + "\t" + (System.currentTimeMillis() - time) / 1000.0);
					bestValue = Math.min(bestValue, ds.getMean());
				}
		log.debug("best value: "+bestValue);
	}
}
