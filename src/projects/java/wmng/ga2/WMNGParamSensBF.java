package wmng.ga2;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.som.grid.Grid2D;
import spawnn.utils.ClusterValidation;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

import context.space.binary_field.SpaceTestDiscrete;
import context.space.binary_field.SpaceTestDiscrete2;

public class WMNGParamSensBF {

	private static Logger log = Logger.getLogger(WMNGParamSensBF.class);

	public static void main(String[] args) {
		final Random r = new Random();

		final int nrNeurons = 10; // mehr neuronen erhöhen unterschiede
		final int maxDist = 5;

		final int maxRfSize = (maxDist + 1) * (maxDist + 2) * 2 - (maxDist + 1) * 4 + 1; // 2d, rook
		final int fa = 0;
		final int[] ga = new int[] { 1, 2 };
		final Dist<double[]> fDist = new EuclideanDist(new int[] { 0 });
		final Dist<double[]> gDist = new EuclideanDist(ga);

		final int T_MAX = 150000;
		int runs = 8;
		int threads = 4;
		final boolean normed = true;

		final List<double[]> samples = DataUtils.readCSV("/home/julian/publications/wmdmng/geographical_systems_v2/data/grid/toroid50x50_1.csv");
		final Map<double[], Map<double[], Double>> dMap = SpaceTestDiscrete.readDistMap(samples, "/home/julian/publications/wmdmng/geographical_systems_v2/data/grid/toroid50x50_1.wtg");
		
		double[] meanBasicNG = new double[maxDist+1];
		{ // basic NG
		long time = System.currentTimeMillis();
		Set<Result> results = new HashSet<Result>();
		try {								
				ExecutorService es = Executors.newFixedThreadPool(threads);
				List<Future<Result>> futures = new ArrayList<Future<Result>>();

				for (int run = 0; run < runs; run++) {
					final int RUN = run;
					futures.add(es.submit(new Callable<Result>() {

						@Override
						public Result call() {

							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] rs = samples.get(r.nextInt(samples.size()));
								neurons.add(Arrays.copyOf(rs, rs.length));
							}

							Sorter<double[]> bg = new DefaultSorter<>(fDist);
							NG ng = new NG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.005, bg);

							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								ng.train((double) t / T_MAX, x);
							}

							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, neurons, bg);
							Map<double[], Set<Grid2D<Boolean>>> rf = SpaceTestDiscrete.getReceptiveFields(samples, dMap, bmus, maxDist, maxRfSize, ga, fa);
															
							Result r = new Result();
							r.bmus = bmus;
							r.qe = SpaceTestDiscrete2.getUncertainty(rf, maxDist, normed);
							return r;
						}
					}));
					}
				es.shutdown();
				for (Future<Result> f : futures)
					results.add(f.get());
		
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		log.debug("took: "+(System.currentTimeMillis()-time)/1000.0+"s");
		
		// calc mean	
		for (Result re : results )
			for (int i = 0; i <= maxDist; i++)
				meanBasicNG[i] += re.qe[i] / results.size();
		log.debug("meanBasicNG: "+Arrays.toString(meanBasicNG));
		}

		/*{ // cng
			long time = System.currentTimeMillis();
			try {
				FileWriter fw = new FileWriter("output/cng_bf.csv");
				fw.write("# runs " + runs + ", maxDist " + maxDist + "\n");
				fw.write("l,dummy");
				for (int i = 0; i <= maxDist; i++)
					fw.write(",dist_" + i);
				fw.write("\n");

				List<double[]> params = new ArrayList<double[]>();
				for (int i = 1; i <= nrNeurons; i++)
					params.add(new double[] { i });
				// params.add( new double[]{10});

				for (final double[] param : params) {
					log.debug(Arrays.toString(param));

					if (param[0] > nrNeurons)
						continue;

					ExecutorService es = Executors.newFixedThreadPool(threads);
					List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

					for (int run = 0; run < runs; run++) {
						final int RUN = run;
						futures.add(es.submit(new Callable<double[]>() {

							@Override
							public double[] call() {

								List<double[]> neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] rs = samples.get(r.nextInt(samples.size()));
									neurons.add(Arrays.copyOf(rs, rs.length));
								}

								Sorter<double[]> bg = new KangasSorter<double[]>(gDist, fDist, (int) param[0]);
								NG ng = new NG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.005, bg);

								for (int t = 0; t < T_MAX; t++) {
									double[] x = samples.get(r.nextInt(samples.size()));
									ng.train((double) t / T_MAX, x);
								}
								Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, neurons, bg);
								Map<double[], Set<Grid2D<Boolean>>> rf = SpaceTestDiscrete.getReceptiveFields(samples, dMap, bmus, maxDist, maxRfSize, ga, fa);
								return SpaceTestDiscrete2.getUncertainty(rf, maxDist, normed);
							}
						}));

					}
					es.shutdown();
					fw.write(param[0] + ",0");
					for (int i = 0; i <= maxDist; i++) {
						double mean = 0;
						for (Future<double[]> f : futures)
							mean += f.get()[i] / runs;
						fw.write("," + mean);
					}
					fw.write("\n");
				}
				fw.close();
			} catch (IOException e) {
				e.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			log.debug("took: " + (System.currentTimeMillis() - time) / 1000.0 + "s");
		}*/
		
		List<double[]> params = new ArrayList<double[]>();
		double steps = 0.05;
		for (double alpha = 0.0; alpha <= 1; alpha += steps, alpha = Math.round(alpha * 10000) / 10000.0 )
			for (double beta = 0.0; beta <= 1; beta += steps, beta = Math.round(beta * 10000) / 10000.0 ) {
				double[] d = new double[] { alpha, beta };
				params.add(d);
			}
					
		/*params.add( new double[]{0.6,0.4} );
		params.add( new double[]{0.4,0.9});
		
		params.add( new double[]{0.5,0.5} );
		params.add( new double[]{0.5,0.9});*/
								
		// alpha = 0: entspricht standard SOM, CDs sind irrelevant
		// alpha = 1: Nur CDs sind relevant
		// beta  = 0: nur die prototypen der nachbarn werden considered
		// beta  = 1: nur die CDs von Nachbarn werden herangezogen
		// problem unterschiede 0.25, 0.5, 0.75 gering, besser 0.1,0.5,0.9 

		long time = System.currentTimeMillis();
		Map<double[], Set<Result>> results = new HashMap<double[], Set<Result>>();
		try {
												
			for (final double[] param : params) {
				log.debug(Arrays.toString(param));

				ExecutorService es = Executors.newFixedThreadPool(threads);
				List<Future<Result>> futures = new ArrayList<Future<Result>>();

				for (int run = 0; run < runs; run++) {
					final int RUN = run;
					futures.add(es.submit(new Callable<Result>() {

						@Override
						public Result call() {

							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] rs = samples.get(r.nextInt(samples.size()));
								double[] d = Arrays.copyOf( rs, rs.length * 2 );
								for (int j = rs.length; j < d.length; j++)
									d[j] = r.nextDouble();
								neurons.add(d);
							}

							Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
							for (double[] d : samples)
								bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

							SorterWMC bg = new SorterWMC(bmuHist, dMap, fDist, param[0], param[1]);
							ContextNG ng = new ContextNG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.005, bg);

							bg.bmuHistMutable = true;
							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								ng.train((double) t / T_MAX, x);
							}
							bg.bmuHistMutable = false;

							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, neurons, bg);
							Map<double[], Set<Grid2D<Boolean>>> rf = SpaceTestDiscrete.getReceptiveFields(samples, dMap, bmus, maxDist, maxRfSize, ga, fa);
							
							Result r = new Result();
							r.bmus = bmus;
							r.qe = SpaceTestDiscrete2.getUncertainty(rf, maxDist, normed);
							return r;
						}
					}));}
				es.shutdown();
				results.put(param, new HashSet<Result>());
				for (Future<Result> f : futures)
					results.get(param).add(f.get());
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		log.debug("took: "+(System.currentTimeMillis()-time)/1000.0+"s");
		
		// calc means
		Map<double[], double[]> means = new HashMap<double[], double[]>();
		for (Entry<double[], Set<Result>> e : results.entrySet()) {
			double[] mean = new double[maxDist+1];
			for (Result re : e.getValue())
				for (int i = 0; i < mean.length; i++)
					mean[i] += re.qe[i] / e.getValue().size();

			means.put(e.getKey(), mean);
		}

		// write means to file
		try {
			FileWriter fw = new FileWriter("output/wmng_bf.csv");
			fw.write("# runs " + runs + ", maxDist " + maxDist + "\n");
			fw.write("alpha,beta");
			for (int i = 0; i <= maxDist; i++)
				fw.write(",dist_" + i);
			fw.write("\n");

			for (Entry<double[], double[]> e : means.entrySet()) {
				fw.write(e.getKey()[0] + ","+e.getKey()[1]);
				for (int i = 0; i <= maxDist; i++)
					fw.write("," + e.getValue()[i]);
				fw.write("\n");
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// write diff-means to file
		try {
			FileWriter fw = new FileWriter("output/wmng_bf_diff.csv");
			fw.write("# runs " + runs + ", maxDist " + maxDist + "\n");
			fw.write("alpha,beta");
			for (int i = 0; i <= maxDist; i++)
				fw.write(",dist_" + i);
			fw.write("\n");

			for (Entry<double[], double[]> e : means.entrySet()) {
				fw.write(e.getKey()[0] + ","+e.getKey()[1]);
				for (int i = 0; i <= maxDist; i++)
					fw.write("," + (meanBasicNG[i] - e.getValue()[i] ) );
				fw.write("\n");
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// calculate mean NMIs
		try {
			FileWriter fw = new FileWriter("output/wmng_bf_nmi.csv");
			fw.write("# runs " + runs + ", maxDist " + maxDist + "\n");
			fw.write("alpha,beta,meanNMI\n");
		
			for( double[] p :params ) {
				List<Result> l = new ArrayList<Result>(results.get(p));
				double meanNMI = 0;
				for( int i = 0; i < l.size(); i++ )
					for( int j = i+1; j < l.size()-1; j++ ) 
						meanNMI += ClusterValidation.getNormalizedMutualInformation(l.get(i).bmus.values(), l.get(j).bmus.values() );
				fw.write(p[0]+","+p[1]+","+(meanNMI/((l.size()*l.size()-1)/2))+"\n" );
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}
