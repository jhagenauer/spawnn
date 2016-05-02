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

import myga.Evaluator;
import myga.GeneticAlgorithm;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.RandomDist;
import spawnn.utils.GraphUtils;

public class GridSearchRegionalization {

	private static Logger log = Logger.getLogger(GridSearchRegionalization.class);

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
				while (samples.size() < 80) {
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
		
		final Dist<double[]> fDist = new EuclideanDist(new int[] { 2 });
		final Dist<double[]> rDist = new RandomDist<double[]>();

		int threads = 3;
		int runs = 3;
		List<Data> data = new ArrayList<Data>();
		while (data.size() < 2)
			data.add(new Data());
		
		final Evaluator<TreeIndividual> evaluator = new WSSEvaluator(fDist);
		final int numCluster = 5;
		//final Evaluator<TreeIndividual> evaluator = new MSTEvaluator(fDist);
		
		GeneticAlgorithm.debug = false;
		for( final int k : new int[]{ 2 } )
		for( final boolean mst : new boolean[]{ true, false } )
		for (final int tournamenSize : new int[] { 2, 3 })
		for (final double recombProb : new double[]{ 0.0, 0.1, 0.2, 0.3 ,0.4 ,0.5, 0.6, 0.7, 0.8, 0.9 }) {
			
			if( mst ) { // only mut cuts
				TreeIndividual.onlyMutCuts = true;
				TreeIndividual.onlyMutTrees = false;
			} else { // mut all
				TreeIndividual.onlyMutCuts = false;
				TreeIndividual.onlyMutTrees = false;
			}
			
				long time = System.currentTimeMillis();
				ExecutorService es = Executors.newFixedThreadPool(threads);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
				GeneticAlgorithm.tournamentSize = tournamenSize;
				GeneticAlgorithm.recombProb = recombProb;
				
				for (final Data d : data) {
					for (int i = 0; i < runs; i++) {
						futures.add(es.submit(new Callable<double[]>() {
							@Override
							public double[] call() throws Exception {
								Random r = new Random();
								List<TreeIndividual> init = new ArrayList<TreeIndividual>();
								while (init.size() < 50) {
									Map<double[], Set<double[]>> tree;
									
									if( mst )
										tree = GraphUtils.getMinimumSpanningTree(d.cm, fDist);
									else
										tree = GraphUtils.getMinimumSpanningTree(d.cm, rDist);
									
									Map<double[], Set<double[]>> cuts = new HashMap<double[], Set<double[]>>();
									int numCuts = 0;
									while (numCuts < numCluster - 1) {
										double[] na = new ArrayList<double[]>(tree.keySet()).get(r.nextInt(tree.keySet().size()));
										double[] nb = new ArrayList<double[]>(tree.get(na)).get(r.nextInt(tree.get(na).size()));

										if (!cuts.containsKey(na) || !cuts.get(na).contains(nb)) {
											if (!cuts.containsKey(na))
												cuts.put(na, new HashSet<double[]>());
											cuts.get(na).add(nb);

											if (!cuts.containsKey(nb))
												cuts.put(nb, new HashSet<double[]>());
											cuts.get(nb).add(na);
											numCuts++;
										}
									}
									if( k == 1 )
										init.add(new TreeIndividual(d.cm, tree, cuts));
									else if( k == 2 )
										init.add(new TreeIndividual2(d.cm, tree, cuts));
								}
								GeneticAlgorithm<TreeIndividual> ga = new GeneticAlgorithm<>(evaluator);
								TreeIndividual bestGA = ga.search(init);
								return new double[] { bestGA.getValue() };
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
				log.info(k + "\t" + mst+"\t"+tournamenSize+"\t"+recombProb+"\t"+ds.getMean() + "\t" + ds.getStandardDeviation()+"\t"+(System.currentTimeMillis()-time)/1000.0);
			}
	}
}
