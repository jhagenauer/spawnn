package rankMeans;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.PowerDecay;

public class RankMeans {

	public static Random r = new Random();
	private static Logger log = Logger.getLogger(RankMeans.class);

	public static Map<double[], Set<double[]>> kMeans(List<double[]> samples, int num, Dist<double[]> aDist,
			Dist<double[]> bDist, int k, double delta) {
		KangasSorter<double[]> sorter = new KangasSorter<>(aDist, bDist, k);
		int length = samples.iterator().next().length;

		Map<double[], Set<double[]>> clusters = null;
		Set<double[]> centroids = new HashSet<double[]>();

		// get num unique(!) indices for centroids
		Set<Integer> indices = new HashSet<Integer>();
		while (indices.size() < num)
			indices.add(r.nextInt(samples.size()));

		for (int i : indices) {
			double[] d = samples.get(i);
			centroids.add(Arrays.copyOf(d, d.length));
		}

		while (true) {
			clusters = new HashMap<double[], Set<double[]>>();
			for (double[] v : centroids)
				// init cluster
				clusters.put(v, new HashSet<double[]>());

			for (double[] s : samples) { // build cluster
				List<double[]> l = new ArrayList<double[]>(clusters.keySet());
				sorter.sort(s, l);
				clusters.get(l.get(0)).add(s);
			}

			boolean converged = true;
			for (double[] c : clusters.keySet()) {
				Collection<double[]> s = clusters.get(c);

				if (s.isEmpty())
					continue;

				// calculate new centroids
				double[] centroid = new double[length];
				for (double[] v : s)
					for (int i = 0; i < v.length; i++)
						centroid[i] += v[i];
				for (int i = 0; i < centroid.length; i++)
					centroid[i] /= s.size();

				// update centroids
				centroids.remove(c);
				centroids.add(centroid);

				if (converged && aDist.dist(c, centroid) > delta && bDist.dist(c, centroid) > delta) {
					// log.debug("not converged: "+aDist.dist(c,
					// centroid)+","+bDist.dist(c, centroid) );
					converged = false;
				}
			}
			if (converged)
				break;
		}
		return clusters;
	}

	public static void main(String[] args) {
		Dist<double[]> aDist = new EuclideanDist(new int[] { 0 });
		Dist<double[]> bDist = new EuclideanDist(new int[] { 1 });
		Sorter<double[]> aSorter = new DefaultSorter<>(aDist);

		int numProto = 10;

		Path file = Paths.get("output/rankMeans.csv");
		try {
			Files.createDirectories(file.getParent()); // create output dir
			Files.deleteIfExists(file);
			Files.createFile(file);
			String s = "method,type,k"; // type geo or feature
			for (int i = 0; i < numProto; i++)
				s += ",proto_" + i;
			s += "\r\n";
			Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		ExecutorService es = Executors.newFixedThreadPool(8);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

		for (int run = 0; run < 250; run++) {
			futures.add(es.submit(new Callable<double[]>() {
				@Override
				public double[] call() throws Exception {

					List<double[]> samples = new ArrayList<double[]>();
					while (samples.size() < 2000)
						samples.add(new double[] { Math.pow(r.nextDouble(), 1), r.nextDouble() });

					// kangas
					for (int k = 1; k <= 10; k++) {
						log.debug("kangas " + k);
						/*Map<double[],Set<double[]>> cluster = kMeans(samples, numProto, aDist, bDist, k, delta); List<double[]> 
						 l = new ArrayList<>(cluster.keySet());*/

						Sorter<double[]> sorter = new KangasSorter<double[]>(aDist, bDist, k);
						List<double[]> l = new ArrayList<>();
						for (int i = 0; i < numProto; i++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							l.add(Arrays.copyOf(x, x.length));
						}
						NG ng = new NG(l, new PowerDecay(5, 0.05), new PowerDecay(0.5, 0.001), sorter);
						for (int t = 0; t < 100000; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / 100000, x);
						}
						aSorter.sort(new double[] { 0 }, l);

						synchronized( this ) {
							try {
								String s = "";
								
								s += "kangas,geo," + k;
								for (double[] d : l)
									s += "," + d[0];
								s += "\r\n";
								
								s += "kangas,feature," + k;
								for (double[] d : l)
									s += "," + d[1];
								s += "\r\n";
								
								Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
					}

					log.debug("weighted");
					for (int k = 0; k <= 100; k++) {
						// log.debug("weighted "+k);
						double p = (double) k / 100;
						Map<Dist<double[]>, Double> m = new HashMap<>();
						m.put(aDist, 1.0 - p);
						m.put(bDist, p);

						/*Map<double[],Set<double[]>> cluster = Clustering.kMeans(samples, numProto, new WeightedDist<>(m), delta ); 
						  List<double[]> l = new ArrayList<double[]>(cluster.keySet());
						 */

						Sorter<double[]> sorter = new DefaultSorter<double[]>(new WeightedDist<>(m));
						List<double[]> l = new ArrayList<>();
						for (int i = 0; i < numProto; i++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							l.add(Arrays.copyOf(x, x.length));
						}
						NG ng = new NG(l, new PowerDecay(5, 0.05), new PowerDecay(0.5, 0.001), sorter);
						for (int t = 0; t < 100000; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / 100000, x);
						}
						aSorter.sort(new double[] { 0 }, l);

						synchronized(this) {
							try {
								String s = "";
								
								s += "weighted,geo," + k;
								for (double[] d : l)
									s += "," + d[0];
								s += "\r\n";
								
								s += "weighted,feature," + k;
								for (double[] d : l)
									s += "," + d[1];
								s += "\r\n";
								
								Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
					}
					return null;
				}
			}));
		}
		es.shutdown();
	}
}
