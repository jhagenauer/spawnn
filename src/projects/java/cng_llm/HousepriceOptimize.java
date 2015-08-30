package cng_llm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
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

import llm.ErrorSorter;
import llm.LLMNG;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class HousepriceOptimize {

	private static Logger log = Logger.getLogger(HousepriceOptimize.class);

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		final DecimalFormat df = new DecimalFormat("00");
		
		final List<double[]> samples = new ArrayList<double[]>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		final List<double[]> desired = new ArrayList<double[]>();

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] { 6, 7 }, new int[] {}, true);
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

		final int da = sdf.names.indexOf("lnp");
		for (double[] d : sdf.samples) {
			if (d[sdf.names.indexOf("time_index")] < 6)
				continue;
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))]; 
					
			samples.add(nd);
			desired.add(new double[]{d[da]});
			geoms.add(sdf.geoms.get(idx));
		}

		final int[] fa = new int[vars.size()-2]; // omit geo-vars
		for( int i = 0; i < fa.length; i++ )
			fa[i] = i+2;
		final int[] ga = new int[] { 0, 1 };

		// ------------------------------------------------------------------------

		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(samples, da); // should not be necessary
		
		final DataFrame resData = DataUtils.readDataFrameFromCSV(new File("output/res.csv"), new int[]{}, true);
		log.debug( resData.names);
		
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		// Ohne ignore ist es etwas besser
		for( final int T_MAX : new int[]{ 20000 } )
		for( final int nrNeurons : new int[]{ 60 } )
		for( final double lInit : new double[]{ nrNeurons })
		for( final double lFinal : new double[]{ 1.0 })	
		for( final double lr1Init : new double[]{ 0.5 })
		for( final double lr1Final : new double[]{ 0.001 })
		for( final double lr2Init : new double[]{ 0.1 })
		for( final double lr2Final : new double[]{ 0.001 })
		for( final boolean ign : new boolean[]{ false } )
		for (int l = 1; l <= 5; l++) {
			final int L = l;

			ExecutorService es = Executors.newFixedThreadPool(3);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 32; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						ErrorSorter errorSorter = new ErrorSorter(samples, desired);
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, errorSorter, L);

						DecayFunction nbRate = new PowerDecay(/*neurons.size()/*/lInit, lFinal);
						
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						LLMNG ng = new LLMNG(neurons, 
								nbRate, lrRate1, 
								nbRate, lrRate2, 
								sorter, fa, 1, ign );
						errorSorter.setLLMNG(ng);

						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samples.size());
							ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
						}
						
						List<double[]> response = new ArrayList<double[]>();
						for (double[] x : samples)
							response.add(ng.present(x));					
						double rmse1 =  Meuse.getRMSE(response, desired);
						
						DefaultSorter<double[]> fSorter = new DefaultSorter<>(fDist);
						ng.setSorter( new KangasSorter<>(gSorter, fSorter, L) );
						List<double[]> response2 = new ArrayList<double[]>();
						for( double[] x : samples )
							response2.add( ng.present(x) );
						double rmse2 =  Meuse.getRMSE(response2, desired);
						
						double sum = 0;
						Map<double[],Set<double[]>> m = NGUtils.getBmuMapping(samples, neurons, sorter);
						for( double[] n : m.keySet() ) {
							double[] coef = ng.matrix.get(n)[0];
							double[] nCoef = new double[coef.length+2];
							for( int i = 0;  i < coef.length; i++ )
								nCoef[i+2] = coef[i];
							
							for( double[] d : m.get(n) ) {
								int idx = samples.indexOf(d);
								sum += Math.pow( fDist.dist(nCoef, resData.samples.get(idx) ), 2);
							}
						}																	
						return new double[] { rmse1, rmse2, Meuse.getR2(response2, desired) };
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
				String fn = "output/result.csv";
				if( firstWrite ) {
					firstWrite = false;
					Files.write(Paths.get(fn), ("t_max,nrNeurons,l,lInit,lFinal,lr1Init,lr1Final,lr2Init,lr2Final,rmse,rmse_sd,rmse2,rmse2_sd,sum,sum_sd\n").getBytes());
				}
				String s = T_MAX+","+nrNeurons+","+l+","+lInit+","+lFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+"";
				for (int i = 0; i < ds.length; i++)
					s += ","+ds[i].getMean()+","+ds[i].getStandardDeviation();
				s += "\n";
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
				System.out.print(s);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
