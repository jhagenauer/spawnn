package cng_houseprice;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
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

import javax.imageio.ImageIO;

import llm.LLMNG;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.Mark;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.rbf.RBF;
import spawnn.utils.ColorBrewerUtil;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class OptimizeHousingLLMCNG {

	private static Logger log = Logger.getLogger(OptimizeHousingLLMCNG.class);

	enum model {
		LLMNG, LINREG, RBF
	};

	public static void main(String[] args) {
		final int T_MAX = 100000;

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

		/*
		 * SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/cng/test2a_nonoise.shp"), true); List<String> vars = new ArrayList<String>(); vars.add("X"); vars.add("Y"); vars.add("VALUE");
		 * 
		 * for (double[] d : sdf.samples) { int idx = sdf.samples.indexOf(d); double[] nd = new double[vars.size()]; for (int i = 0; i < nd.length; i++) nd[i] = d[sdf.names.indexOf(vars.get(i))]; samples.add(nd); desired.add(new double[] { d[sdf.names.indexOf("CLASS")] }); geoms.add(sdf.geoms.get(idx)); }
		 */

		int[] fa = new int[vars.size() - 2];
		for (int i = 0; i < fa.length; i++)
			fa[i] = i + 2;
		final int[] ga = new int[] { 0, 1 };

		DataUtils.zScoreColumns(samples, fa);

		// PCA
		int nrComponents = 4;
		List<double[]> ns = DataUtils.removeColumns(samples, ga);
		ns = DataUtils.reduceDimensionByPCA(ns, nrComponents);
		for (int k = 0; k < ns.size(); k++) {
			double[] d = ns.get(k);
			double[] nd = new double[ga.length + d.length];
			for (int i = 0; i < ga.length; i++)
				nd[i] = d[ga[i]];
			for (int i = 0; i < nrComponents; i++)
				nd[i + ga.length] = d[i];
			samples.set(k, nd);
		}
		final int[] fFa = new int[nrComponents];
		for (int i = 0; i < nrComponents; i++)
			fFa[i] = i + ga.length;

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fFa);

		boolean firstWrite = true;
		double[][] bestParams = new double[3][];
		double bestMean[] = new double[bestParams.length];
		for (final model m : new model[]{ model.LINREG, model.LLMNG } )
			for (final double a : new double[] { 9 })
				for (double k = 1; k <= a; k++) {
					final int K = (int) k;

					ExecutorService es = Executors.newFixedThreadPool(4);
					List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

					for (int i = 0; i < 100; i++) {
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

								Sorter<double[]> sorter = new KangasSorter<double[]>(gDist, fDist, K);

								List<double[]> responseVal = new ArrayList<double[]>();
								Map<double[], Set<double[]>> mapping = null;
								if (m == model.LLMNG) {
									List<double[]> neurons = new ArrayList<double[]>();
									for (int i = 0; i < a; i++) {
										double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
										neurons.add(Arrays.copyOf(d, d.length));
									}
									LLMNG ng = new LLMNG(neurons, a / 2, 0.1, 0.5, 0.111, a / 2, 0.1, 0.1, 0.001, sorter, fFa, 1);

									for (int t = 0; t < T_MAX; t++) {
										int j = r.nextInt(samplesTrain.size());
										ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
									}
									mapping = NGUtils.getBmuMapping(samplesVal, ng.getNeurons(), sorter);

									for (double[] x : samplesVal)
										responseVal.add(ng.present(x));

								} else if (m == model.LINREG) {
									List<double[]> neurons = new ArrayList<double[]>();
									for (int i = 0; i < a; i++) {
										double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
										neurons.add(Arrays.copyOf(d, d.length));
									}
									NG ng = new NG(neurons, a / 2, 0.1, 0.5, 0.001, sorter);

									for (int t = 0; t < T_MAX; t++)
										ng.train((double) t / T_MAX, samples.get(r.nextInt(samples.size())));
									mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter);

									List<Integer> toIgnore = new ArrayList<Integer>();
									for (int i : ga)
										toIgnore.add(i);

									double[] y = new double[samplesTrain.size()];
									double[][] x = new double[samplesTrain.size()][];
									int l = 0;
									for (double[] d : samplesTrain) {
										int idx = samplesTrain.indexOf(d);
										y[l] = desiredTrain.get(idx)[0];
										x[l] = new double[d.length - toIgnore.size() + neurons.size() - 1];
										int j = 0;
										for (int i = 0; i < d.length; i++) {
											if (toIgnore.contains(i))
												continue;
											x[l][j++] = d[i];
										}
										// add dummy variable
										for (int k = 1; k < neurons.size(); k++)
											if (mapping.get(neurons.get(k)).contains(d))
												x[l][j + k - 1] = 1;
										l++;
									}

									OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
									ols.setNoIntercept(false);
									ols.newSampleData(y, x);
									double[] beta = ols.estimateRegressionParameters();

									for (double[] d : samplesVal) {
										double[] nd = new double[d.length - toIgnore.size() + neurons.size() - 1];
										int j = 0;
										for (int i = 0; i < d.length; i++) {
											if (toIgnore.contains(i))
												continue;
											nd[j++] = d[i];
										}
										// add dummy variable
										for (int k = 1; k < neurons.size(); k++)
											if (mapping.get(neurons.get(k)).contains(d))
												nd[j + k - 1] = 1;

										double ps = beta[0]; // intercept at beta[0]
										for (int k = 1; k < beta.length; k++)
											ps += beta[k] * nd[k - 1];

										responseVal.add(new double[] { ps });
									}

								} else if (m == model.RBF) {
									List<double[]> neurons = new ArrayList<double[]>();
									for (int i = 0; i < a; i++) {
										double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
										neurons.add(Arrays.copyOf(d, d.length));
									}
									NG ng = new NG(neurons, a / 2, 0.1, 0.5, 0.001, sorter);

									for (int t = 0; t < T_MAX; t++)
										ng.train((double) t / T_MAX, samples.get(r.nextInt(samples.size())));
									mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter);

									Map<double[], Double> hidden = new HashMap<double[], Double>();
									// min plus overlap
									for (double[] c : mapping.keySet()) {
										double d = Double.MAX_VALUE;
										for (double[] n : mapping.keySet())
											if (c != n)
												d = Math.min(d, fDist.dist(c, n)) * 1.1;
										hidden.put(c, d);
									}

									RBF rbf = new RBF(hidden, 1, fDist, 0.05);
									for (int i = 0; i < T_MAX; i++) {
										int j = r.nextInt(samples.size());
										rbf.train(samples.get(j), desired.get(j));
									}

									for (double[] x : samplesVal)
										responseVal.add(rbf.present(x));
								}

								double rmse = Meuse.getRMSE(responseVal, desiredVal);
								if (Double.isNaN(rmse))
									rmse = Double.MAX_VALUE;

								return new double[] { rmse, Math.pow(Meuse.getPearson(responseVal, desiredVal), 2.0), DataUtils.getMeanQuantizationError(mapping, fDist) };
							}
						}));
					}
					es.shutdown();

					double[] mean = new double[bestMean.length];
					for (Future<double[]> ff : futures) {
						try {
							double[] ee = ff.get();
							for (int i = 0; i < mean.length; i++)
								mean[i] += ee[i] / futures.size();
						} catch (InterruptedException ex) {
							ex.printStackTrace();
						} catch (ExecutionException ex) {
							ex.printStackTrace();
						}
					}
					String s = m + "," + a + "," + k;
					for (double d : mean)
						s += "," + d;
					log.debug(s);
					s += "\n";

					try {
						String fn = "output/result.csv";
						if (firstWrite) {
							Files.write(Paths.get(fn), "model,neurons,k,rmse,r2,qe\n".getBytes());
							firstWrite = false;
						} 
						Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
	}

	public static void geoDrawValues(List<Geometry> geoms, List<Double> values, CoordinateReferenceSystem crs, String fn) {
		SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
		typeBuilder.setName("data");
		typeBuilder.setCRS(crs);
		typeBuilder.add("the_geom", geoms.get(0).getClass());

		SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
		Map<Geometry, Double> m = new HashMap<Geometry, Double>();
		for (int i = 0; i < geoms.size(); i++)
			m.put(geoms.get(i), values.get(i));

		Map<Geometry, Color> colMap = ColorBrewerUtil.valuesToColors(m, ColorMode.Blues);
		Set<Color> cols = new HashSet<Color>(colMap.values());

		try {
			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();

			ReferencedEnvelope mapBounds = mc.getMaxBounds();
			for (Color c : cols) {
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				for (Geometry g : colMap.keySet()) {
					if (!c.equals(colMap.get(g)))
						continue;
					featureBuilder.set("the_geom", g);
					features.add(featureBuilder.buildFeature("" + features.size()));
				}
				Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, c);
				FeatureLayer fl = new FeatureLayer(features, SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null))));
				mc.addLayer(fl);
				mapBounds.expandToInclude(fl.getBounds());
			}

			GTRenderer renderer = new StreamingRenderer();
			renderer.setMapContent(mc);

			Rectangle imageBounds = null;
			
			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			int imageWidth = 2000;
			imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));

			BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
			Graphics2D gr = image.createGraphics();
			renderer.paint(gr, imageBounds, mapBounds);

			ImageIO.write(image, "png", new FileOutputStream(fn));
			image.flush();
			mc.dispose();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
