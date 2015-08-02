package growingCNG;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class GrowingCNG_Housing_Thread {

	private static Logger log = Logger.getLogger(GrowingCNG_Housing_Thread.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 20000;

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] { 6, 7 }, new int[] {}, true);

		final List<double[]> samples = new ArrayList<double[]>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		final List<double[]> desired = new ArrayList<double[]>();

		List<String> vars = new ArrayList<String>();
		vars.add("xco");
		vars.add("yco");
		vars.add("lnarea_tot");
		vars.add("lnarea_plo");
		vars.add("attic_dum");
		vars.add("cellar_dum");
		vars.add("cond_house_3");
		vars.add("heat_3");
		vars.add("bath_3");
		vars.add("garage_3");
		vars.add("terr_dum");
		vars.add("age_num");
		vars.add("time_index");
		vars.add("zsp_alq_09");
		vars.add("gem_kauf_i");
		vars.add("gem_abi");
		vars.add("gem_alter_");
		vars.add("ln_gem_dic");

		for (double[] d : sdf.samples) {
			if (d[sdf.names.indexOf("time_index")] < 6)
				continue;
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))];
			samples.add(nd);
			desired.add(new double[] { d[sdf.names.indexOf("lnp")] });
			geoms.add(sdf.geoms.get(idx));
		}

		final int[] fa = new int[] { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
		final int[] ga = new int[] { 0, 1 };
		DataUtils.zScoreColumns(samples, fa);

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		boolean first = true;

		/* Was beeinflußt error-calc:
		 * - lrB und lrN, denn langsame adaptation macht initialisierung wichtiger
		 * - lambda, denn jede Einfügung geschieht an error-stelle
		 */
		
		// warum kleiner lernerate auf längere sicht nicht besser? Evtl. weil lrN zu viel disrupted?
		for (final double lrB : new double[] { 0.04 }) // 0.0005 konvergiert nicht/kaum wenn add dabei ist
			for (final double lrN : new double[] { 0.0, lrB/100 })
				for (final int lambda : new int[] { 300 })
					for (final int aMax : new int[] { 100 })
						for( final double alpha : new double[]{ 0.5 } )
							for( final double beta : new double[]{ 0.00005 } )
						for (final int distMode : new int[] { 0 }) {

							ExecutorService es = Executors.newFixedThreadPool(1);
							List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

							for (int i = 0; i < 64; i++) {
								futures.add(es.submit(new Callable<double[]>() {

									@Override
									public double[] call() throws Exception {

										List<double[]> neurons = new ArrayList<double[]>();
										for (int i = 0; i < 2; i++) {
											double[] d = samples.get(r.nextInt(samples.size()));
											neurons.add(Arrays.copyOf(d, d.length));
										}

										double ratio = 0.0;
										GrowingCNG ng = new GrowingCNG(neurons, lrB, lrN, gDist, fDist, ratio, aMax, lambda, alpha, beta);
										ng.distMode = distMode;
										ng.samples = samples;
										
										int t = 0;
										List<Double> qes = new ArrayList<Double>();
										while( t < T_MAX ) { 
											double[] x = samples.get(r.nextInt(samples.size())); 
											ng.train(t++, x); 
											/*if (t % 100 == 0) {
												Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), new DefaultSorter<double[]>(fDist));
												double qe = DataUtils.getMeanQuantizationError(mapping, fDist);
												qes.add(qe);
											}*/
											/*if( t % 100 == 0 )
												qes.add( ng.bErrorSum );*/
										}
										
										/*double qe = Double.MAX_VALUE;
										while (qe > 3.34) {
											double[] x = samples.get(r.nextInt(samples.size()));
											ng.train(t++, x);
											if (t % 200 == 0) {
												Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), new KangasSorter<double[]>(gDist, fDist, (int) Math.ceil(ng.getNeurons().size() * ratio)));
												qe = DataUtils.getMeanQuantizationError(mapping, fDist);
											}
										}*/

										
										Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), new DefaultSorter<double[]>(fDist)); 
										int used = 0; 
										for( Set<double[]> s : mapping.values() ) 
											if( !s.isEmpty() ) 
												used++; 
										double[] r = new double[] { t, ng.getNeurons().size(), used, ng.getConections().size(), DataUtils.getMeanQuantizationError(mapping, fDist), ng.aErrorSum, ng.bErrorSum  };
										
										/*double[] r = new double[qes.size()];
										for (int i = 0; i < qes.size(); i++)
											r[i] = qes.get(i);*/
										
										return r;
									}
								}));
							}
							es.shutdown();

							DescriptiveStatistics ds[] = null;
							for (Future<double[]> ff : futures) {
								try {
									double[] ee = ff.get();
									if (ds == null) {
										ds = new DescriptiveStatistics[ee.length];
										for (int i = 0; i < ee.length; i++)
											ds[i] = new DescriptiveStatistics();
									}
									for (int i = 0; i < ee.length; i++)
										ds[i].addValue(ee[i]);
								} catch (InterruptedException ex) {
									ex.printStackTrace();
								} catch (ExecutionException ex) {
									ex.printStackTrace();
								}
							}

							try {
								FileWriter fw = new FileWriter("output/result.csv",!first);
								if( first ) {
									first = false;
									fw.write("params");
									for( int i = 0; i < ds.length; i++ )
										fw.write(",t_"+i);
									fw.write("\n");
								}
								fw.write(lrB + ";" + lrN + ";" + lambda + ";" + aMax + ";" + distMode );
								for (int i = 0; i < ds.length; i++)
									fw.write(","+ds[i].getMean() );
								fw.write("\n");
								fw.close();
							} catch (IOException e) {
								e.printStackTrace();
							}
							System.out.print(lrB + ";" + lrN + ";" + lambda + ";" + aMax + ";" + alpha + ";" + beta + ";" + distMode + ",");
							for (int i = 0; i < ds.length && i < 10; i++) {
								System.out.print(ds[i].getMean() + ",");
							}
							System.out.println();
						}
	}
}
