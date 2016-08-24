package chowClustering;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import chowClustering.ChowClustering.ClusterResult;
import landCon.LandCon;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.GraphUtils;
import spawnn.utils.SpatialDataFrame;

public class GWR_Clustering {

	private static Logger log = Logger.getLogger(GWR_Clustering.class);

	public static void main(String[] args) {
		int threads = 3;

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/publications/landcon/output/merged_dat_gwr.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		Map<double[], Set<double[]>> cm = GraphUtils.deriveQueenContiguitiyMap(samples, geoms, false);

		int[] fa = new int[] { 0, 4, 5, 6, 7, 9 };
		int[] fa2 = new int[] { 23, 24, 25, 26, 27, 28 };
		int ta = 12;

		DataUtils.transform(samples, new int[] { 0 }, Transform.sqrt);
		DataUtils.transform(samples, new int[] { 4 }, Transform.log);

		Path file = Paths.get("output/gwr.txt");
		try {
			Files.deleteIfExists(file);
			Files.createFile(file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		List<Entry<List<Integer>, List<Integer>>> cvList10 = SupervisedUtils.getCVList(10, 10, samples.size());

		// ------------------------------------------------------------------------------------------

		for (HierarchicalClusteringType hct : new HierarchicalClusteringType[] { HierarchicalClusteringType.ward, HierarchicalClusteringType.average_linkage }) {
			Clustering.r.setSeed(0);

			ClusterResult bestAIC = null, bestCV10 = null;

			double[] aics;
			double[] errorsCV10;
			
			String method = "gwr_"+hct;
			
			//List<TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, new EuclideanDist(fa2), hct);
			List<TreeNode> tree = LandCon.getHierarchicalClusterTree(samples, cm, new EuclideanDist(fa2), hct, threads);

			List<Integer> nrCl = new ArrayList<>();
			for (int i = 4; i < 200; i += 4)
				nrCl.add(i);

			errorsCV10 = new double[nrCl.size()];
			aics = new double[nrCl.size()];
			for (int i = 0; i < nrCl.size(); i++) {
				List<Set<double[]>> ct = Clustering.cutTree(tree, nrCl.get(i));

				// CV 10
				{
					ExecutorService es = Executors.newFixedThreadPool(threads);
					List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
					for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList10) {
						futures.add(es.submit(new Callable<double[]>() {
							@Override
							public double[] call() throws Exception {
								List<double[]> samplesTrain = new ArrayList<double[]>();
								for (int k : cvEntry.getKey())
									samplesTrain.add(samples.get(k));

								List<double[]> samplesVal = new ArrayList<double[]>();
								for (int k : cvEntry.getValue())
									samplesVal.add(samples.get(k));

								return new double[] { Math.sqrt( ChowClustering.getSumOfSquares( ChowClustering.getResidualsLM(ct, samplesTrain, samplesVal, fa, ta)) / samplesVal.size()) };
							}
						}));
					}
					es.shutdown();

					SummaryStatistics ss = new SummaryStatistics();
					for (Future<double[]> f : futures)
						try {
							ss.addValue(f.get()[0]);
						} catch (InterruptedException | ExecutionException e) {
							e.printStackTrace();
						}
					errorsCV10[i] = ss.getMean();
				}

				aics[i] = SupervisedUtils.getAICc(ChowClustering.getSumOfSquares(ChowClustering.getResidualsLM(ct, samples, samples, fa, ta)) / samples.size(), ct.size() * (fa.length + 1), samples.size());

				if (bestAIC == null || aics[i] < bestAIC.cost)
					bestAIC = new ClusterResult(-1, aics[i], ct, "aic_" + method);

				if (bestCV10 == null || errorsCV10[i] < bestCV10.cost)
					bestCV10 = new ClusterResult(-1, errorsCV10[i], ct, "cv10_" + method);

			}
			
			try {
				String sErrorCV10 = "rmse_cv10," + method + "," + Arrays.toString(errorsCV10) + "\r\n";
				String sAIC = "aic," + method + "," + Arrays.toString(aics) + "\r\n";

				Files.write(file, sErrorCV10.replace("[", "").replace("]", "").getBytes(), StandardOpenOption.APPEND);
				Files.write(file, sAIC.replace("[", "").replace("]", "").getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e) {
				e.printStackTrace();
			}

			for (ClusterResult cr : new ClusterResult[] { bestAIC, bestCV10 }) {
				log.info(cr);
				Drawer.geoDrawCluster(cr.cluster, samples, geoms, "output/" + cr.method + "_" + cr.cluster.size() + "_" + cr.cost + ".png", true);
			}
		}
	}
}
