package cng_llm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
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

import llm.ErrorSorter;
import llm.LLMNG;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class SpatialHetBacaoOptimize {

	private static Logger log = Logger.getLogger(SpatialHetBacaoOptimize.class);

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		final DecimalFormat df = new DecimalFormat("00");

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("/home/julian/git/spawnn/output/bacao.csv"), new int[] { 0, 1 }, new int[] {}, true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		final List<double[]> desired = new ArrayList<double[]>();

		for (double[] d : samples)
			desired.add(new double[] { d[4] });

		final int[] fa = new int[] { 2, 3 };
		final int[] ga = new int[] { 0, 1 };
		final int T_MAX = 20000;

		// ------------------------------------------------------------------------

		//DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(samples, 4); // should not be necessary

		final Map<Integer, Set<double[]>> ref = new HashMap<Integer, Set<double[]>>();
		for (double[] d : samples) {
			int c = (int) d[5];
			if (!ref.containsKey(c))
				ref.put(c, new HashSet<double[]>());
			ref.get(c).add(d);
		}

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		// Ohne ignore ist es etwas besser
		for( final boolean ignore : new boolean[]{false} )
		for (int l = 1; l <= 6; l++) {
			final int L = l;

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 32; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < 60; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						ErrorSorter errorSorter = new ErrorSorter(samples, desired);
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, errorSorter, L);

						DecayFunction nbRate = new PowerDecay(neurons.size()/3, 1);
						DecayFunction lrRate = new PowerDecay(0.5, 0.005);
						LLMNG ng = new LLMNG(neurons, 
								nbRate, lrRate, 
								nbRate, lrRate, 
								sorter, fa, 1, ignore );
						errorSorter.setLLMNG(ng);

						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samples.size());
							ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
						}
						Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter);

						List<double[]> response = new ArrayList<double[]>();
						for (double[] x : samples)
							response.add(ng.present(x));

						Map<double[], double[]> geoCoefs = new HashMap<double[], double[]>();
						for (double[] n : ng.getNeurons()) {
							double[] cg = new double[2 + fa.length];
							cg[0] = n[ga[0]];
							cg[1] = n[ga[1]];
							for (int i = 0; i < fa.length; i++)
								cg[2 + i] = ng.matrix.get(n)[0][i];
							geoCoefs.put(n, cg);
						}

						// CHL of coefs
						Map<Connection, Integer> conns = new HashMap<Connection, Integer>();
						for (double[] x : samples) {
							sorter.sort(x, ng.getNeurons());
							List<double[]> bmuList = ng.getNeurons();

							Connection c = new Connection(geoCoefs.get(bmuList.get(0)), geoCoefs.get(bmuList.get(1)));
							if (!conns.containsKey(c))
								conns.put(c, 1);
							else
								conns.put(c, conns.get(c) + 1);
						}

						// graph clustering
						Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
						for (Connection c : conns.keySet()) {
							double[] a = c.getA();
							double[] b = c.getB();

							if (!cm.containsKey(a))
								cm.put(a, new HashSet<double[]>());
							cm.get(a).add(b);
							if (!cm.containsKey(b))
								cm.put(b, new HashSet<double[]>());
							cm.get(b).add(a);
						}
						Map<Set<double[]>, TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, fDist, Clustering.HierarchicalClusteringType.ward);
						List<Set<double[]>> coefCluster = Clustering.cutTree(tree, 3);

						List<Set<double[]>> cluster = new ArrayList<Set<double[]>>();
						for (Set<double[]> s : coefCluster) {
							Set<double[]> c = new HashSet<double[]>();
							for (double[] n : mapping.keySet())
								if (s.contains(geoCoefs.get(n)))
									c.addAll(mapping.get(n));
							cluster.add(c);
						}
						
						DefaultSorter<double[]> fSorter = new DefaultSorter<>(fDist);
						ng.setSorter( new KangasSorter<>(gSorter, fSorter, L) );
						List<double[]> response2 = new ArrayList<double[]>();
						for( double[] x : samples )
							response2.add( ng.present(x) );
						double rmse2 =  Meuse.getRMSE(response2, desired);

						return new double[] { 
								Meuse.getRMSE(response, desired), 
								Math.pow(Meuse.getPearson(response, desired), 2), 
								ClusterValidation.getNormalizedMutualInformation(cluster, ref.values()),
								rmse2
						};
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
					Files.write(Paths.get(fn), ("ignore,"
							+ "rmse,rmse_sd,r2,r2_sd,nmi,nmi_sd"
							+ "\n").getBytes());
				}
				String s = ignore+"";
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
