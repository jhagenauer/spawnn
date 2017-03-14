package chowClustering;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

import chowClustering.ChowClustering.PreCluster;
import chowClustering.ChowClustering.StructChangeTestMode;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.ColorBrewer;
import spawnn.utils.ColorUtils.ColorClass;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering_AIC {

	private static Logger log = Logger.getLogger(ChowClustering_AIC.class);

	public static int STRUCT_TEST = 0, P_VALUE = 1, DIST = 2, MIN_OBS = 3, PRECLUST = 4, PRECLUST_OPT = 5,
			PRECLUST_OPT2 = 6;

	public static double best = Double.MAX_VALUE;
	public static LinearModel bestLm = null;

	public static void main(String[] args) {

		int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
		log.debug("Threads: " + threads);

		File data = new File("data/gemeinden_gs2010/gem_dat.shp");
		// File data = new File("gem_dat.shp");
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(data, new int[] { 1, 2 }, true);

		int[] ga = new int[] { 3, 4 };
		int[] fa = new int[] { 7, 8, 9, 10, 19, 20 };
		int[] faPred = new int[] { 12, 13, 14, 15, 19, 21 };
		int ta = 18; // bldUpRt

		for (int i = 0; i < fa.length; i++)
			log.debug("fa " + i + ": " + sdf.names.get(fa[i]));
		log.debug("ta: " + ta + "," + sdf.names.get(ta));

		Dist<double[]> gDist = new EuclideanDist(ga);
		Map<double[], Set<double[]>> cm = GeoUtils.getContiguityMap(sdf.samples, sdf.geoms, false, false);
		Map<double[], Map<double[], Double>> wcm = GeoUtils.contiguityMapToDistanceMap(cm);
		GeoUtils.rowNormalizeMatrix(wcm);

		{
			LinearModel lm = new LinearModel(sdf.samples, fa, ta, false);
			List<Double> pred = lm.getPredictions(sdf.samples, fa);
			double mse = SupervisedUtils.getMSE(pred, sdf.samples, ta);
			log.debug("lm aic: " + SupervisedUtils.getAICc_GWMODEL(mse, fa.length + 1, sdf.samples.size())); // lm aic: -61856.98209268832
		}

		Path file = Paths.get("output/regioclust.csv");
		try {
			Files.createDirectories(file.getParent()); // create output dir
			Files.deleteIfExists(file);
			Files.createFile(file);
			String s = "method,cluster,aic\r\n";
			Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		Path tabFile = Paths.get("output/regioclust_tab.csv");
		try {
			Files.createDirectories(tabFile.getParent()); 
			Files.deleteIfExists(tabFile);
			Files.createFile(tabFile);
			
			String s = "method,cluster,aic,rss,r2,moran,pValue";
			for( int i = 0; i < fa.length; i++ )
				s += ","+sdf.names.get(fa[i])+"_15";
			s += ",Intercept_15";
			
			for( int i = 0; i < fa.length; i++ )
				s += ","+sdf.names.get(fa[i])+"_30";
			s += ",Intercept_30\r\n";
			
			Files.write(tabFile, s.getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		List<Object[]> params = new ArrayList<>();
		for (int l : new int[] { 8, 9, 10, 11, 12 }) {
			params.add(new Object[] { StructChangeTestMode.ResiSimple, 1.0, gDist, l, PreCluster.ward, 1, 1 });
			//params.add(new Object[] { StructChangeTestMode.ResiSimple, 1.0, gDist, l, PreCluster.kmeans, 1700, 1});
		}

		Collections.sort(params, new Comparator<Object[]>() {
			@Override
			public int compare(Object[] o1, Object[] o2) {
				return Integer.compare((int) o1[MIN_OBS], (int) o2[MIN_OBS]);
			}
		});

		for (Object[] param : params) {
			Clustering.r.setSeed(0);

			bestLm = null;

			String method = Arrays.toString(param);
			final double pValue = (double) param[P_VALUE];
			log.debug((params.indexOf(param) + 1) + "/" + params.size() + "," + method);

			List<Future<LinearModel>> futures = new ArrayList<>();
			ExecutorService es = Executors.newFixedThreadPool(threads);

			List<TreeNode> bestCurLayer = null;
			double bestWss = Double.POSITIVE_INFINITY;

			for (int i = 0; i < (int) param[PRECLUST_OPT2]; i++) {

				List<TreeNode> curLayer = ChowClustering.getInitCluster(sdf.samples, cm, (PreCluster) param[PRECLUST], (int) param[PRECLUST_OPT], gDist, (int) param[MIN_OBS], threads);
				curLayer = Clustering.cutTree(curLayer, 1);
				List<Set<double[]>> cluster = Clustering.treeToCluster(curLayer);

				double wss = ClusterValidation.getWithinClusterSumOfSuqares(cluster, gDist);
				if (bestCurLayer == null || wss < bestWss) {
					bestCurLayer = curLayer;
					bestWss = wss;
				}
			}

			Map<TreeNode, Set<TreeNode>> ncm = ChowClustering.getCMforCurLayer(bestCurLayer, cm);
			List<TreeNode> tree = ChowClustering.getFunctionalClusterinTree(bestCurLayer, ncm, fa, ta, (ChowClustering.StructChangeTestMode) param[STRUCT_TEST], pValue, threads);

			int minClust = tree.size();
			for (int i = minClust; i <= (pValue == 1.0 ? Math.min(bestCurLayer.size(), 250) : minClust); i++) {
				final int nrCluster = i;
				futures.add(es.submit(new Callable<LinearModel>() {
					@Override
					public LinearModel call() throws Exception {
						List<Set<double[]>> ct = Clustering.treeToCluster(Clustering.cutTree(tree, nrCluster));
						LinearModel lm = new LinearModel(sdf.samples, ct, fa, ta, false);
						double mse = SupervisedUtils.getMSE(lm.getPredictions(sdf.samples, fa), sdf.samples, ta);
						double aic = SupervisedUtils.getAICc_GWMODEL(mse, ct.size() * (fa.length + 1), sdf.samples.size());
						//double aic = SupervisedUtils.getBIC(mse, ct.size() * (fa.length + 1), sdf.samples.size());

						synchronized (this) {

							if (bestLm == null || aic < best) {
								best = aic;
								bestLm = lm;
							}

							try {
								String s = "";
								s += "\"" + method + "\"," + ct.size() + "," + aic + "\r\n";
								Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
							} catch (IOException e) {
								e.printStackTrace();
							}
						}

						return null;
					}
				}));
			}

			es.shutdown();
			try {
				es.awaitTermination(24, TimeUnit.HOURS);
			} catch (InterruptedException e1) {
				e1.printStackTrace();
			}
			System.gc();

			{ // ----- output best
				
				LinearModel lm = bestLm;
				List<Set<double[]>> ct = lm.cluster;
				double mse = SupervisedUtils.getMSE(lm.getPredictions(sdf.samples, fa), sdf.samples, ta);
				double aic = SupervisedUtils.getAICc_GWMODEL(mse, ct.size() * (fa.length + 1), sdf.samples.size());
				double r2 = SupervisedUtils.getR2(lm.getPredictions(sdf.samples, fa), sdf.samples, ta);
				
				Map<double[], Double> values = new HashMap<>();
				for (int i = 0; i < sdf.samples.size(); i++)
					values.put(sdf.samples.get(i), lm.getResiduals().get(i));
				double[] moran = GeoUtils.getMoransIStatistics(wcm, values);
				
				log.info("####### " + Arrays.toString(param) + " ########");
				log.info("#cluster: " + lm.cluster.size());
				log.info("rss: " + lm.getRSS());
				log.info("r2: " + r2 );
				log.info("aicc: " + aic);
				log.info("mse: " + mse);
				log.info("wss: " + ClusterValidation.getWithinClusterSumOfSuqares(ct, gDist));
				log.info("moran: " + Arrays.toString(moran));
				
				try {
					String s = "\"" + method + "\"," + ct.size() + "," + aic +","+lm.getRSS()+","+r2+","+moran[0]+","+moran[4];
					for( int i = 0; i < fa.length+1; i++ ) {
						DescriptiveStatistics ds = new DescriptiveStatistics();
						for( int j = 0; j < lm.getCluster().size(); j++ )
							ds.addValue( lm.getBeta(j)[i] );
						double q1 = ds.getPercentile(25);
						double q3 = ds.getPercentile(75);
												
						int outlier15 = 0;
						for( int j = 0; j < lm.getCluster().size(); j++ ) {
							double v = lm.getBeta(j)[i];
							if( v < q1-1.5*(q3-q1) || v > q3+1.5*(q3-q1) ) {
								outlier15++;
							}
						}
						s += "," + (double)outlier15/ct.size();
						
						int outlier30 = 0;
						for( int j = 0; j < lm.getCluster().size(); j++ ) {
							double v = lm.getBeta(j)[i];
							if( v < q1-3.0*(q3-q1) || v > q3+3.0*(q3-q1) ) {
								outlier30++;
							}
						}
						s += "," + (double)outlier30/ct.size();
					}
					s += "\r\n";
					Files.write(tabFile, s.getBytes(), StandardOpenOption.APPEND);
				} catch (IOException e) {
					e.printStackTrace();
				}

				List<Double> predictions = lm.getPredictions(sdf.samples, faPred);
				List<double[]> l = new ArrayList<double[]>();
				for (double[] d : sdf.samples) {
					double[] ns = new double[3 + fa.length + 1];

					int i = sdf.samples.indexOf(d);
					ns[0] = lm.getResiduals().get(i);
					ns[1] = predictions.get(i);

					for (int j = 0; j < lm.cluster.size(); j++) {
						if (!lm.cluster.get(j).contains(d))
							continue;

						ns[2] = j; // cluster

						double[] beta = lm.getBeta(j);
						for (int k = 0; k < beta.length; k++)
							ns[3 + k] = beta[k];
						break;
					}
					l.add(ns);
				}

				double hiFam = Double.NEGATIVE_INFINITY;
				double pHiFam = Double.NaN;

				String[] names = new String[3 + fa.length + 1];
				names[0] = "residual";
				names[1] = "prdction";
				names[2] = "cluster";
				for (int i = 0; i < fa.length; i++)
					names[3 + i] = sdf.names.get(fa[i]);
				names[names.length - 1] = "Intrcpt";

				DataUtils.writeShape(l, sdf.geoms, names, sdf.crs, "output/" + method + ".shp");

				List<String> dissNames = new ArrayList<>();
				for (int i = 0; i < fa.length; i++)
					dissNames.add(sdf.names.get(fa[i]));
				dissNames.add("Intrcpt");

				for (int i = 0; i < fa.length; i++)
					dissNames.add("p_" + sdf.names.get(fa[i]));
				dissNames.add("p_Intrcpt");

				dissNames.add("numObs");
				dissNames.add("cluster");
				dissNames.add("mse");

				List<double[]> dissSamples = new ArrayList<>();
				List<Geometry> dissGeoms = new ArrayList<>();
				for (Set<double[]> set : lm.cluster) {
					int idx = lm.cluster.indexOf(set);

					// multipolys to list of polys
					List<Polygon> polys = new ArrayList<>();
					for (double[] d : set) {
						int idx2 = sdf.samples.indexOf(d);

						MultiPolygon mp = (MultiPolygon) sdf.geoms.get(idx2);
						for (int i = 0; i < mp.getNumGeometries(); i++)
							polys.add((Polygon) mp.getGeometryN(i));
					}

					while (true) {
						int bestI = -1, bestJ = -1;
						for (int i = 0; i < polys.size() - 1 && bestI < 0; i++) {
							for (int j = i + 1; j < polys.size(); j++) {

								if (!polys.get(i).intersects(polys.get(j)))
									continue;

								Geometry is = polys.get(i).intersection(polys.get(j));
								if (is instanceof Point || is instanceof MultiPoint)
									continue;

								bestI = i;
								bestJ = j;
								break;
							}
						}
						if (bestI < 0)
							break;

						Polygon a = polys.remove(bestJ);
						Polygon b = polys.remove(bestI);
						polys.add((Polygon) a.union(b));
					}
					Geometry union = new GeometryFactory().createMultiPolygon(polys.toArray(new Polygon[] {}));

					List<Double> dl = new ArrayList<>();
					double[] beta = lm.getBeta(idx);
					for (double d : beta)
						dl.add(d);

					double[] se = lm.getBetaStdError(idx);
					if (set.size() > beta.length) {
						TDistribution td = new TDistribution(set.size() - beta.length);
						for (int i = 0; i < beta.length; i++) {
							double tValue = beta[i] / se[i];
							double pv = 2 * (td.cumulativeProbability(-Math.abs(tValue)));
							dl.add(pv);

							if (i == 0 && beta[i] > hiFam) {
								hiFam = beta[0];
								pHiFam = pv;
							}
						}
					} else
						dl.add(Double.NaN);

					dl.add((double) set.size());
					dl.add((double) idx);
					dl.add(lm.getRSS(idx) / set.size());

					// dl (list) to da (array)
					double[] da = new double[dl.size()];
					for (int i = 0; i < dl.size(); i++)
						da[i] = dl.get(i);

					dissSamples.add(da);
					dissGeoms.add(union);

					// DataUtils.writeCSV("output/"+idx+".csv", new
					// ArrayList<>(s), sdf.names.toArray(new String[]{}));
				}
				log.debug(hiFam + "," + pHiFam);
				DataUtils.writeShape(dissSamples, dissGeoms, dissNames.toArray(new String[] {}), sdf.crs, "output/" + method + "_diss.shp");
				Drawer.geoDrawValues(dissGeoms, dissSamples, fa.length + 3, sdf.crs, ColorBrewer.Set3, ColorClass.Equal, "output/" + method + "_cluster.png");
			} // ----
		}
	}

}
