package llm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class LLMNG_Housing_Optimize {

	private static Logger log = Logger.getLogger(LLMNG_Housing_Optimize.class);

	public static void main(String[] args) {
		boolean firstWrite = true;
		final int T_MAX = 10000;

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

		for (int nrNeurons = 29; nrNeurons < 30; nrNeurons++)
			for (int l = 1; l <= nrNeurons; l++) {
				final int N = nrNeurons, L = l;

				ExecutorService es = Executors.newFixedThreadPool(4);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int i = 0; i < 32; i++) {
					futures.add(es.submit(new Callable<double[]>() {

						@Override
						public double[] call() throws Exception {
							Random r = new Random();
							int samplesSize = samples.size();
							List<double[]> samplesTrain = new ArrayList<double[]>(samples);
							List<double[]> desiredTrain = new ArrayList<double[]>(desired);

							List<double[]> samplesVal = new ArrayList<double[]>();
							List<double[]> desiredVal = new ArrayList<double[]>();
							while (samplesVal.size() < 0.3 * samplesSize) {
								int idx = r.nextInt(samplesTrain.size());
								samplesVal.add(samplesTrain.remove(idx));
								desiredVal.add(desiredTrain.remove(idx));
							}
							
							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < N; i++) {
								double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
								neurons.add(Arrays.copyOf(d, d.length));
							}

							ErrorSorter errorSorter = new ErrorSorter(samplesTrain, desiredTrain);
							Sorter<double[]> sorter = new KangasSorter<>(new DefaultSorter<>(gDist), errorSorter, L);
							LLMNG ng = new LLMNG(neurons, neurons.size(), 0.1, 0.5, 0.005, neurons.size(), 0.1, 0.1, 0.005, sorter, fa, 1);
							errorSorter.setLLMNG(ng);

							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samplesTrain.size());
								ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
							}
							
							Sorter[] s = new Sorter[]{
								new DefaultSorter<>(gDist),
								new DefaultSorter<>(fDist),
								new KangasSorter<>(new DefaultSorter<>(gDist), new DefaultSorter<>(fDist), L)
							};
							double[] rmses = new double[s.length];
							for( int i = 0; i < rmses.length; i++ ) {
								ng.setSorter(s[i]);
								List<double[]> responseVal = new ArrayList<double[]>();							
								for (double[] x : samplesVal)
									responseVal.add(ng.present(x));
								rmses[i] = Meuse.getRMSE(responseVal, desiredVal);
							}
							return rmses;
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
						Files.write(Paths.get(fn), "neurons,l,s1,s2,s3\n".getBytes());
					}
					String s = nrNeurons+","+l;
					for (int i = 0; i < ds.length; i++)
						s += ","+ds[i].getMean();
					s += "\n";
					Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
					System.out.print(s);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
	}
}
