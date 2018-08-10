package context.space;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.ContextNG;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class SpaceTest9 {

	private static Logger log = Logger.getLogger(SpaceTest9.class);

	public static void main(String[] args) {

		Random r = new Random();
		int T_MAX = 150000;
		int rcpFieldSize = 80;

		File file = new File("data/redcap/Election/election2004.shp");
		SpatialDataFrame sd = DataUtils.readShapedata(file, new int[] {}, true);
		List<double[]> samples = sd.samples;
		List<Geometry> geoms = sd.geoms;
		Map<double[], Set<double[]>> ctg = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");

		// build dist matrix and add coordinates to samples
		Map<double[], Map<double[], Double>> distMap = new HashMap<double[], Map<double[], Double>>();
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p1 = geoms.get(i).getCentroid();

			distMap.put(d, new HashMap<double[], Double>());
			for (double[] nb : ctg.get(d)) {
				int j = samples.indexOf(nb);

				if (i == j)
					continue;

				Point p2 = geoms.get(j).getCentroid();
				distMap.get(d).put(nb, p1.distance(p2));
			}

			// ugly, but easy
			d[0] = p1.getX();
			d[1] = p1.getY();

			// normed coordinates
			d[2] = p1.getX();
			d[3] = p1.getY();
		}

		final int[] ga = new int[] { 0, 1 };
		final int[] gaNormed = new int[] { 2, 3 };

		final int fa = 7;
		final Dist<double[]> fDist = new EuclideanDist(new int[] { fa });
		final Dist<double[]> gDist = new EuclideanDist(ga);

		DataUtils.zScoreColumn(samples, fa);
		DataUtils.zScoreGeoColumns(samples, gaNormed, gDist);
		final Dist<double[]> normedGDist = new EuclideanDist(gaNormed);

		// build ctg-dist-Matrix
		Map<double[], Map<double[], Double>> cMap = new HashMap<double[], Map<double[], Double>>();
		for (double[] d : ctg.keySet()) {
			Map<double[], Double> dists = new HashMap<double[], Double>();
			for (double[] nb : ctg.get(d))
				if (d != nb)
					dists.put(nb, 1.0);

			cMap.put(d, dists);
		}

		// write ctg-dist-matrix as weight matrix
		/*
		 * try { BufferedWriter bw = new BufferedWriter(new
		 * FileWriter("output/election2004_queen.wtg"));
		 * //bw.write("id1,id2,dist\n"); for (int i = 0; i < samples.size();
		 * i++) { for (int j = 0; j < samples.size(); j++) { double[] a =
		 * samples.get(i); double[] b = samples.get(j); if (cMap.containsKey(a)
		 * && cMap.get(a).containsKey(b) && a != b )
		 * bw.write(i+","+j+","+1.0+"\n"); } } bw.close(); } catch (Exception e)
		 * { } System.exit(1);
		 */

		// get knns
		Map<double[], List<double[]>> knns = new HashMap<double[], List<double[]>>();
		for (double[] x : samples) {
			List<double[]> sub = new ArrayList<double[]>();
			while (sub.size() <= rcpFieldSize) { // sub.size() must be larger than cLength!

				double[] minD = null;
				for (double[] d : samples)
					if (!sub.contains(d) && (minD == null || gDist.dist(d, x) < gDist.dist(minD, x)))
						minD = d;
				sub.add(minD);
			}
			knns.put(x, sub);
		}
		log.debug("knn build.");

		DecimalFormat df = new DecimalFormat("0000");

		for (int run = 0; run < 1000; run++) {
			File outdir = new File("output/" + df.format(run));
			outdir.mkdir();

			try {

				FileWriter fw = new FileWriter(outdir + "/performance.txt");

				Map<double[], Set<double[]>> wngBmus = null;
				{ // weighted ng
					for (double w : new double[] { 0.3 }) {
						log.debug("weighted ng " + w);

						Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
						map.put(fDist, 1 - w);
						map.put(normedGDist, w);
						Dist<double[]> wDist = new WeightedDist<double[]>(map);

						DefaultSorter<double[]> bg = new DefaultSorter<double[]>(wDist);
						NG ng = new NG(9, 9.0 / 2, 0.01, 0.5, 0.005, samples.get(0).length, bg);

						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / T_MAX, x);
						}

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
						wngBmus = bmus;
						double[] qe = SpaceTest.getQuantizationErrorOld(samples, bmus, fDist, rcpFieldSize, knns);
						double sum = 0;
						for( int i = 1; i < 20; i++ )
							sum += qe[i];

						DataUtils.writeCSV(outdir+"/wng_neurons_" + w + ".csv", ng.getNeurons(), null);
						Drawer.geoDrawCluster(bmus.values(), samples, geoms, outdir+"/wng_" + w + ".png", true);
						
						fw.write("wng");
						fw.write( DataUtils.getMeanQuantizationError(wngBmus, fDist)+","+qe[0]+","+sum+"\n" );
					}
				}

				Map<double[], Set<double[]>> geoSomBmus = null;
				{ // geosom
					for (int l : new int[] { 2 }) { // 2
						log.debug("geosom " + l);

						spawnn.som.bmu.BmuGetter<double[]> bg = new spawnn.som.bmu.KangasBmuGetter<double[]>(normedGDist, fDist, l);
						Grid2D<double[]> grid = new Grid2DHex<double[]>(3, 3);
						SomUtils.initRandom(grid, samples);

						SOM som = new SOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 1)), new LinearDecay(1.0, 0.0), grid, bg);
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							som.train((double) t / T_MAX, x);
						}

						Map<GridPos, Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);

						geoSomBmus = new HashMap<double[], Set<double[]>>();
						for (GridPos p : bmus.keySet())
							geoSomBmus.put(grid.getPrototypeAt(p), bmus.get(p));

						double[] qe = SpaceTest.getQuantizationErrorOld(samples, bmus, fDist, rcpFieldSize, knns);
						double sum = 0;
						for( int i = 1; i < 20; i++ )
							sum += qe[i];

						DataUtils.writeCSV(outdir+"/geosom_neurons_" + l + ".csv", grid.getPrototypes(), null);
						Drawer.geoDrawCluster(bmus.values(), samples, geoms, outdir+"/GeoSOM_" + l + ".png", true);
						
						fw.write("geosom");
						fw.write( DataUtils.getMeanQuantizationError(geoSomBmus, fDist)+","+qe[0]+","+sum+"\n" );
					}
				}

				Map<double[], Set<double[]>> cngBmus = null;
				{ // cng
					for (int l : new int[] { 6 }) {
						log.debug("cng: " + l);

						KangasSorter<double[]> bg = new KangasSorter<double[]>(normedGDist, fDist, l);
						NG ng = new NG(9, 9.0 / 2, 0.01, 0.5, 0.005, samples.get(0).length, bg);

						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / T_MAX, x);
						}

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
						cngBmus = bmus;

						double[] qe = SpaceTest.getQuantizationErrorOld(samples, bmus, fDist, rcpFieldSize, knns);
						double sum = 0;
						for( int i = 1; i < 20; i++ )
							sum += qe[i];

						DataUtils.writeCSV(outdir+"/cng_neurons.csv", ng.getNeurons(), null);
						Drawer.geoDrawCluster(bmus.values(), samples, geoms, outdir+"/cng_" + l + ".png", true);
						
						fw.write("cng");
						fw.write( DataUtils.getMeanQuantizationError(cngBmus, fDist)+","+qe[0]+","+sum+"\n" );
					}
				}

				Map<double[], Set<double[]>> wmdmngBmus = null;
				{ // WMNG
					List<double[]> settings = new ArrayList<double[]>();
					settings.add(new double[] { -1, 0.85, 0.75 });

					for (double[] s : settings) {
						double band = s[0];
						double alpha = s[1];
						double beta = s[2];

						log.debug("WMNG " + alpha + "," + beta + "," + band);

						final Map<double[], Map<double[], Double>> dMap;
						if (band < 0) {
							dMap = new HashMap<double[], Map<double[], Double>>();
							for (double[] d : ctg.keySet()) {
								Map<double[], Double> dists = new HashMap<double[], Double>();
								for (double[] nb : ctg.get(d))
									dists.put(nb, 1.0);

								double n = dists.size();
								for (double[] nb : ctg.get(d))
									dists.put(nb, 1.0 / n);

								dMap.put(d, dists);
							}
						} else {
							dMap = SpaceTest.getDistMatrix(samples, gDist, band);
						}

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < 9; i++) {
							double[] rs = samples.get(r.nextInt(samples.size()));
							double[] d = Arrays.copyOf(rs, rs.length * 2);
							for (int j = rs.length; j < d.length; j++)
								d[j] = r.nextDouble();
							neurons.add(d);
						}

						Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
						for (double[] d : samples)
							bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

						SorterWMC bg = new SorterWMC(bmuHist, dMap, fDist, alpha, beta);
						ContextNG ng = new ContextNG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.005, bg);

						bg.bmuHistMutable = true;
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / T_MAX, x);
						}
						bg.bmuHistMutable = false;

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
						wmdmngBmus = bmus;

						double[] qe = SpaceTest.getQuantizationErrorOld(samples, bmus, fDist, rcpFieldSize, knns);
						double sum = 0;
						for( int i = 1; i < 20; i++ )
							sum += qe[i];

						DataUtils.writeCSV(outdir+"/wmdng_neurons.csv", ng.getNeurons(), null);
						Drawer.geoDrawCluster(bmus.values(), samples, geoms, outdir+"/wmng_" + band + "_" + alpha + "_" + beta + ".png", true);
						
						fw.write("wmng");
						fw.write( DataUtils.getMeanQuantizationError(cngBmus, fDist)+","+qe[0]+","+sum+"\n" );

					}
				}

				// match prototypes and write shp-file
				{
					List<List<double[]>> protos = new ArrayList<List<double[]>>();
					protos.add(new ArrayList<double[]>()); // wng
					protos.add(new ArrayList<double[]>()); // geosom
					protos.add(new ArrayList<double[]>()); // cng
					protos.add(new ArrayList<double[]>()); // wmdmng
					for (double[] d : wngBmus.keySet())
						protos.get(0).add(d);
					for (double[] d : geoSomBmus.keySet())
						protos.get(1).add(d);
					for (double[] d : cngBmus.keySet())
						protos.get(2).add(d);
					for (double[] d : wmdmngBmus.keySet())
						protos.get(3).add(d);

					List<int[]> x = new ArrayList<int[]>();
					for (int i = 0; i < protos.size(); i++) {
						int[] j = new int[protos.get(i).size()];
						for (int k = 0; k < j.length; k++)
							j[k] = k;
						x.add(j);
					}
					double xValue = Double.MAX_VALUE;

					double maxT = 10; // temperature
					int maxI = 1200;
					int maxJ = 10;
					for (double t = maxT; t > 0.0001; t = t * 0.95) { // 0.001,
																		// 0.92
						int j = 0;
						for (int i = 0; i < maxI;) {

							// copy
							List<int[]> y = new ArrayList<int[]>();
							for (int[] k : x)
								y.add(Arrays.copyOf(k, k.length));

							// mutate/swap
							int kIdx = r.nextInt(y.size());
							int[] k = y.get(kIdx);
							int idxA = r.nextInt(k.length);
							int idxB = r.nextInt(k.length);
							int tmp = k[idxA];
							k[idxA] = k[idxB];
							k[idxB] = tmp;

							// slow, needs improvement
							double yValue = 0;
							int numCluster = y.get(0).length;
							for (int l = 0; l < numCluster; l++) { // for all
																	// (9)
																	// clusters
								Set<double[]> union = new HashSet<double[]>();

								for (int m = 0; m < numCluster; m++) {
									if (y.get(0)[m] == l)
										union.addAll(wngBmus.get(protos.get(0).get(m)));
									if (y.get(1)[m] == l)
										union.addAll(geoSomBmus.get(protos.get(1).get(m)));
									if (y.get(2)[m] == l)
										union.addAll(cngBmus.get(protos.get(2).get(m)));
									if (y.get(3)[m] == l)
										union.addAll(wmdmngBmus.get(protos.get(3).get(m)));
								}
								yValue += union.size();
							}
							yValue /= 7000;

							// l muß größer sein, je höher t und je besser y
							// ist.
							// ist y = x, ist l = 1!;
							double l = Math.exp((xValue - yValue) / t);
							if (yValue < xValue || r.nextDouble() < l) {
								x = y;

								xValue = yValue;
								i++;
								// log.debug("het: "+xValue+", "+t+","+i);
							} else if (maxJ < ++j) {
								j = 0;
								i++;
							}
						}
						log.debug("t: " + t + "->" + xValue);
					}
					fw.write("union-size: " + (int) (xValue * 7000)+"\n");

					log.debug("writing shape-file...");
					List<double[]> s = new ArrayList<double[]>();
					for (double[] d : samples) {
						double[] dd = new double[4];

						for (int i = 0; i < protos.get(0).size(); i++)
							if (wngBmus.get(protos.get(0).get(i)).contains(d))
								dd[0] = x.get(0)[i];

						for (int i = 0; i < protos.get(1).size(); i++)
							if (geoSomBmus.get(protos.get(1).get(i)).contains(d))
								dd[1] = x.get(1)[i];

						for (int i = 0; i < protos.get(2).size(); i++)
							if (cngBmus.get(protos.get(2).get(i)).contains(d))
								dd[2] = x.get(2)[i];

						for (int i = 0; i < protos.get(3).size(); i++)
							if (wmdmngBmus.get(protos.get(3).get(i)).contains(d))
								dd[3] = x.get(3)[i];

						s.add(dd);
					}

					DataUtils.writeShape(s, geoms, new String[] { "WNG", "GeoSOM", "CNG", "WMNG" }, sd.crs, outdir+"/election_clusters.shp");
					log.debug("done.");

				}
				
				fw.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

	}
}
