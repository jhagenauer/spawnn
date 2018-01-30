package gwr.ga;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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
import com.vividsolutions.jts.geom.GeometryFactory;

import heuristics.GeneticAlgorithm;
import nnet.SupervisedUtils;
import spawnn.utils.ColorBrewer;
import spawnn.utils.ColorUtils.ColorClass;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWRGeneticAlgorithm<T extends GAIndividual<T>> {

	private static Logger log = Logger.getLogger(GWRGeneticAlgorithm.class);
	private final static Random r = new Random(0);
	int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);;

	public static int tournamentSize = 2;
	public static double recombProb = 0.7;
	public static boolean elitist = false;

	public T search(List<T> init, CostCalculator<T> cc) {

		List<T> gen = new ArrayList<T>(init);
		Map<T, Double> costs = new HashMap<T, Double>(); // cost cache
		for (T i : init)
			costs.put(i, cc.getCost(i));

		T best = null;
		double bestCost = Double.MAX_VALUE;

		int noImpro = 0;
		int parentSize = init.size();
		int offspringSize = parentSize * 2;

		int maxK = 50000;
		int k = 0;
		while (k < maxK && noImpro < 200) {

			// check best and increase noImpro
			noImpro++;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (T cur : gen) {
				if (best == null || costs.get(cur) < bestCost) {
					best = cur;
					noImpro = 0;
					bestCost = costs.get(cur);
				}
				ds.addValue(costs.get(cur));
			}
			if (noImpro == 0 || k % 100 == 0) {
				log.debug(k + "," + ds.getMin() + "," + ds.getMean() + "," + ds.getMax() + ","
						+ ds.getStandardDeviation());
				
				if( k % 100 == 0 )
					log.debug( ((GWRIndividual)best).getChromosome() );
			}

			// SELECT NEW GEN/POTENTIAL PARENTS
			// elite
			Collections.sort(gen, new Comparator<GAIndividual<T>>() {
				@Override
				public int compare(GAIndividual<T> g1, GAIndividual<T> g2) {
					return Double.compare(costs.get(g1), costs.get(g2));
				}
			});

			List<T> elite;
			if( !elitist )
				elite = new ArrayList<T>();
			else
				elite = new ArrayList<>(gen.subList(0, Math.max(1, (int) Math.round( gen.size()*0.05 ) ) ) );

			// SELECT PARENT
			gen.removeAll(elite); // elite always parent
			List<T> selected = new ArrayList<T>(elite); 
			while (selected.size() < parentSize) {
				T i = tournament(gen, tournamentSize, costs);
				selected.add(i);
			}
			gen = selected;

			// GENERATE OFFSPRING
			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<SimpleEntry<T, Double>>> futures = new ArrayList<Future<SimpleEntry<T, Double>>>();

			for (int i = 0; i < offspringSize; i++) {
				final T a = gen.get(r.nextInt(gen.size()));
				final T b = gen.get(r.nextInt(gen.size()));

				futures.add(es.submit(new Callable<SimpleEntry<T, Double>>() {
					@Override
					public SimpleEntry<T, Double> call() throws Exception {
						T child;
						if (r.nextDouble() < recombProb)
							child = a.recombine(b);
						else
							child = a;
						T mutChild = child.mutate();

						return new SimpleEntry<T, Double>(mutChild, cc.getCost(mutChild));
					}
				}));
			}
			es.shutdown();

			gen.clear();
			for (Future<SimpleEntry<T, Double>> f : futures) {
				try {
					SimpleEntry<T, Double> e = f.get();
					gen.add(e.getKey());
					costs.put(e.getKey(), e.getValue());
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			gen.addAll(elite);
			costs.keySet().retainAll(gen);

			k++;
		}
		log.info("no impro " + k + ", cost " + bestCost + ", rate " + bestCost / (k - noImpro));
		return best;
	}

	// tournament selection
	private T tournament(List<T> gen, int k, Map<T, Double> costs) {
		List<T> ng = new ArrayList<T>();

		double sum = 0;
		for (int i = 0; i < k; i++) {
			T in = gen.get(r.nextInt(gen.size()));
			ng.add(in);
			sum += costs.get(in);
		}

		Collections.sort(ng, new Comparator<T>() {
			@Override
			public int compare(T g1, T g2) {
				return Double.compare(costs.get(g1), costs.get(g2));
			}
		});

		// deterministic
		return ng.get(0);
	}

	private T binaryProbabilisticTournament(List<T> gen, double prob, Map<T, Double> costs) {
		Random r = new Random();
		T a = gen.get(r.nextInt(gen.size()));
		T b = gen.get(r.nextInt(gen.size()));

		if (costs.get(b) < costs.get(a)) {
			T tmp = a;
			a = b;
			b = tmp;
		}
		if (r.nextDouble() < prob)
			return a;
		else
			return b;
	}

	// roulette wheel selection
	private T rouletteWheelSelect(List<T> gen, Map<T, Double> costs) {
		double sum = 0;
		for (T in : gen)
			sum += costs.get(in);

		Random r = new Random();
		double v = r.nextDouble();

		double a = 0, b = 0;
		for (int j = 0; j < gen.size(); j++) {
			a = b;
			b = (sum - costs.get(gen.get(j))) / sum + b;
			if (a <= v && v <= b || j + 1 == gen.size() && a <= v)
				return gen.get(j);
		}
		return null;
	}

	// stochastic universal sampling
	private List<T> sus(List<T> gen, int n, Map<T, Double> costs) {
		List<T> l = new ArrayList<T>();
		Collections.sort(gen, new Comparator<T>() {
			@Override
			public int compare(T g1, T g2) {
				return Double.compare(costs.get(g1), costs.get(g2));
			}
		});

		double sum = 0;
		for (T in : gen)
			sum += costs.get(in);

		// intervals
		double ivs[] = new double[gen.size() + 1];
		ivs[0] = 0.0f;
		for (int j = 0; j < ivs.length - 1; j++)
			ivs[j + 1] = sum - costs.get(gen.get(j)) + ivs[j];

		double start = r.nextDouble() * sum / n;
		for (int i = 0; i < n; i++) {
			double v = start + i * sum / n;
			// binary search of v
			int first = 0, last = ivs.length - 1;
			while (true) {
				int mid = first + (last - first) / 2;

				if (last - first <= 1) {
					l.add(gen.get(mid));
					break;
				}
				if (ivs[first] <= v && v <= ivs[mid])
					last = mid;
				else if (ivs[mid] <= v && v <= ivs[last])
					first = mid;
			}
		}
		return l;
	}

	public static void main(String[] args) {

		GeometryFactory gf = new GeometryFactory();
		GWKernel kernel = GWKernel.bisquare;
		boolean adaptive = true;
		double minBw = 8;
		int bwInit = -1;
		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 3 };
		int ta = 4;

		int pointsPerRow = (int)Math.pow(2, 4);
		List<double[]> samples = new ArrayList<double[]>(BuildTestData.createSpDepTestData(pointsPerRow));
		List<Geometry> geoms = new ArrayList<Geometry>();
		for (double[] d : samples)
			geoms.add(gf.createPoint(new Coordinate(d[ga[0]], d[ga[1]])));
		log.debug(samples.size());
		DataUtils.writeCSV("output/spDat.csv", samples, new String[] { "long", "lat", "beta", "x1", "y" });

		Map<double[], Integer> idxMap = new HashMap<double[], Integer>();
		for (int i = 0; i < samples.size(); i++)
			idxMap.put(samples.get(i), i);

		double delta = 0.0000001;
		Map<Integer, Set<Integer>> cmI = new HashMap<Integer, Set<Integer>>();
		for (int i = 0; i < samples.size(); i++) {
			double[] a = samples.get(i);
			Set<Integer> s = new HashSet<Integer>();
			for (int j = 0; j < samples.size(); j++) {
				double[] b = samples.get(j);
				if (Math.abs(a[ga[0]] - b[ga[0]]) <= delta + 1.0 / pointsPerRow
						&& Math.abs(a[ga[1]] - b[ga[1]]) <= delta + 1.0 / pointsPerRow) // expects 1/pointsPerRow spacing
					s.add(j);
			}
			cmI.put(i, s);
		}
		GWRIndividual.cmI = cmI;

		Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
		for (int i : cmI.keySet()) {
			Set<double[]> s = new HashSet<double[]>();
			for (int j : cmI.get(i))
				s.add(samples.get(j));
			cm.put(samples.get(i), s);
		}

		Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.contiguityMapToDistanceMap(cm));
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, samples.size());

		GWRCostCalculator ccAICc = new GWRIndividualCostCalculator_AICc(samples, fa, ga, ta, kernel, adaptive, minBw);

		{
			log.info("Search global bandwidth/j");
			double bestGlobalACIc = Double.MAX_VALUE;
			for (int j = 2; j < pointsPerRow; j++) {
				List<Double> bw = new ArrayList<Double>();
				for (int i = 0; i < samples.size(); i++)
					bw.add((double) j);
				GWRIndividual ind = new GWRIndividual(bw, 0, Double.MAX_VALUE, 0);
				
				double aicc = ccAICc.getCost(ind);
								
				log.debug(j+","+aicc);
				if (aicc < bestGlobalACIc) {
					bestGlobalACIc = aicc;
					bwInit = j;
					log.info("bw " + j);
					log.info("basic GWR AICc: " + aicc);
				}
			}
		}

		List<Object[]> params = new ArrayList<Object[]>();
		for( int ts : new int[]{2,3,4,6,8,10} )
			for( boolean mutMode : new boolean[]{ true, false })
				for( boolean meanRecomb : new boolean[]{true, false} )
					for( boolean elitist : new boolean[]{true, false } )
						for( double recomProb: new double[]{ 0.5,0.6,0.7,0.8,0.9})
							for( double sd : new double[]{ 0.5,1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 5.0 })
								params.add( new Object[]{ts,mutMode,meanRecomb,elitist,recomProb,sd,false});
		
		Collections.shuffle(params);
		
		for (Object[] p : params) {	
			
			GWRGeneticAlgorithm.tournamentSize = (int) p[0];	
			GWRGeneticAlgorithm.elitist = (boolean)p[3];
			GWRGeneticAlgorithm.recombProb = (double) p[4];
			
			GWRIndividual.mutationMode = (boolean) p[1];
			GWRIndividual.meanRecomb = (boolean) p[2];
			double sd = (double) p[5];
			boolean intInd = (boolean)p[6];
			
			log.info(params.indexOf(p)+","+Arrays.toString(p));

			List<GWRIndividual> init = new ArrayList<GWRIndividual>();
			while (init.size() < 25) {
				List<Double> bandwidth = new ArrayList<>();
				while (bandwidth.size() < samples.size())
					bandwidth.add( (double)Math.round( bwInit + r.nextGaussian() * 4 ) ); 
				if( !intInd )
					init.add(new GWRIndividual(bandwidth, sd, bandwidth.size(), minBw ));
				else
					init.add(new GWRIndividual_Int(bandwidth, sd, bandwidth.size(), minBw ));
			}

			GWRGeneticAlgorithm<GWRIndividual> gen = new GWRGeneticAlgorithm<GWRIndividual>();
			GWRIndividual result = (GWRIndividual) gen.search(init, ccAICc);
			Map<double[], Double> resultBw = ccAICc.getSpatialBandwidth(result);
			
			double aicc = ccAICc.getCost(result);
			
			log.info("result GWR AICc: " + aicc);

			List<double[]> r = new ArrayList<double[]>();
			Map<double[], Double> r2 = new HashMap<double[], Double>();
			for (int i = 0; i < samples.size(); i++) {
				double[] d = samples.get(i);
				r.add(new double[] { d[ga[0]], d[ga[1]], d[2], resultBw.get(d) });
				r2.put(d, resultBw.get(d) );
			}
			log.info("moran: " + Arrays.toString(GeoUtils.getMoransIStatistics(dMap, r2)));

			String s = "output/result_AICc_" + params.indexOf(p)+","+Arrays.toString(p) + "_" + aicc;
			DataUtils.writeCSV(s + ".csv", r, new String[] { "long", "lat", "b", "radius" });
			Drawer.geoDrawValues(geoms, r, 3, null, ColorBrewer.Blues, ColorClass.Equal, s + ".png");
		}
	}
}
