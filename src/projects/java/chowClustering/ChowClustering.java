package chowClustering;

import java.io.File;
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
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.distribution.FDistribution;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.PrecisionModel;
import com.vividsolutions.jts.operation.union.UnaryUnionOp;
import com.vividsolutions.jts.precision.GeometryPrecisionReducer;

import landCon.LandCon;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.GraphUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering {

	private static Logger log = Logger.getLogger(ChowClustering.class);

	static boolean debug = false;

	static List<TreeNode> getHierarchicalClusterTree(List<TreeNode> leafLayer, Map<TreeNode, Set<TreeNode>> cm, int[] fa, int ta, HierarchicalClusteringType hct, Dist<double[]> dist, int minSize, StructChangeTestMode sctm, int threads) {

		class FlatSet<T> extends HashSet<T> {
			private static final long serialVersionUID = -1960947872875758352L;
			public int hashCode = 0;

			@Override
			public boolean add(T t) {
				hashCode += t.hashCode();
				return super.add(t);
			}

			@Override
			public boolean addAll(Collection<? extends T> c) {
				hashCode += c.hashCode();
				return super.addAll(c);
			}

			@Override
			public int hashCode() {
				return hashCode;
			}
		}

		List<TreeNode> tree = new ArrayList<>();
		Map<TreeNode, Set<double[]>> curLayer = new HashMap<>();

		Map<TreeNode, Double> ssCache = new ConcurrentHashMap<TreeNode, Double>();
		Map<TreeNode, Map<TreeNode, Double>> unionCache = new ConcurrentHashMap<>();

		int age = 0;
		for (TreeNode tn : leafLayer) {
			age = Math.max(age, tn.age);
			tree.add(tn);

			Set<double[]> content = Clustering.getContents(tn);
			curLayer.put(tn, content);

			ssCache.put(tn, DataUtils.getSumOfSquares(content, dist));
		}

		// copy of connected map
		final Map<TreeNode, Set<TreeNode>> connected = new HashMap<TreeNode, Set<TreeNode>>();
		if (cm != null)
			for (Entry<TreeNode, Set<TreeNode>> e : cm.entrySet())
				connected.put(e.getKey(), new HashSet<TreeNode>(e.getValue()));

		boolean firstPhase = true;

		while (curLayer.size() > 1) {

			TreeNode c1 = null, c2 = null;
			double sMin = Double.POSITIVE_INFINITY;

			List<TreeNode> cl = new ArrayList<>(curLayer.keySet());

			if (firstPhase) {
				boolean b = true;
				for (Set<double[]> s : curLayer.values())
					if (s.size() < minSize)
						b = false;
				if (b) {
					log.debug("start second/chow phase with " + curLayer.size());
					ssCache.clear();
					unionCache.clear();
					firstPhase = false;
				}
			}
			final boolean SEC_PHASE = !firstPhase;

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			for (int t = 0; t < threads; t++) {
				final int T = t;

				futures.add(es.submit(new Callable<double[]>() {
					@Override
					public double[] call() throws Exception {
						int c1 = -1, c2 = -1;
						double minCost = Double.POSITIVE_INFINITY;

						for (int i = T; i < cl.size() - 1; i += threads) {
							TreeNode l1 = cl.get(i);

							for (int j = i + 1; j < cl.size(); j++) {
								TreeNode l2 = cl.get(j);

								if (!connected.containsKey(l1) || !connected.get(l1).contains(l2)) // disjoint
									continue;

								double cost = Double.NaN;
								List<double[]> s1 = new ArrayList<>(curLayer.get(l1));
								List<double[]> s2 = new ArrayList<>(curLayer.get(l2));

								if (SEC_PHASE) {
									cost = testStructChange(getX(s1, fa, true), getY(s1, ta), getX(s2, fa, true), getY(s2, ta), sctm);
								} else {
									if (hct == HierarchicalClusteringType.ward) {
										if (!unionCache.containsKey(l1) || !unionCache.get(l1).containsKey(l2)) {
											if (!unionCache.containsKey(l1))
												unionCache.put(l1, new HashMap<TreeNode, Double>());

											Set<double[]> union = new HashSet<>(s1);
											union.addAll(s2);
											unionCache.get(l1).put(l2, DataUtils.getSumOfSquares(union, dist));
										}
										cost = unionCache.get(l1).get(l2) - (ssCache.get(l1) + ssCache.get(l2));

									} else if (HierarchicalClusteringType.single_linkage == hct) {
										cost = Double.MAX_VALUE;
										for (double[] d1 : s1)
											for (double[] d2 : s2)
												cost = Math.min(cost, dist.dist(d1, d2));
									} else if (HierarchicalClusteringType.complete_linkage == hct) {
										cost = Double.MIN_VALUE;
										for (double[] d1 : s1)
											for (double[] d2 : s2)
												cost = Math.max(cost, dist.dist(d1, d2));
									} else if (HierarchicalClusteringType.average_linkage == hct) {
										cost = 0;
										for (double[] d1 : s1)
											for (double[] d2 : s2)
												cost += dist.dist(d1, d2);
										cost /= (curLayer.get(l1).size() * curLayer.get(l2).size());
									}
								}

								if (limitString != null && limitString.equals(LIMIT) && s1.size() + s2.size() > minSize)
									cost += (s1.size() + s2.size() - minSize) * 1000;
								else if (limitString != null && limitString.equals(LIMIT2) && s1.size() + s2.size() > minSize)
									cost += 10000;

								if (cost < minCost) {
									c1 = i;
									c2 = j;
									minCost = cost;
								}
							}
						}
						return new double[] { c1, c2, minCost };
					}
				}));
			}
			es.shutdown();
			try {
				for (Future<double[]> f : futures) {
					double[] d = f.get();

					if (d[2] < sMin) {
						c1 = cl.get((int) d[0]);
						c2 = cl.get((int) d[1]);
						sMin = d[2];
					}
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}

			if (c1 == null && c2 == null) { // no connected clusters present
											// anymore
				log.debug("only non-connected clusters present! " + curLayer.size());
				return tree;
			}

			// create merge node, remove c1,c2
			Set<double[]> union = new FlatSet<double[]>();
			union.addAll(curLayer.remove(c1));
			union.addAll(curLayer.remove(c2));

			TreeNode mergeNode = new TreeNode(++age, sMin);
			mergeNode.children = Arrays.asList(new TreeNode[] { c1, c2 });
			ssCache.remove(c1);
			ssCache.remove(c2);

			// add nodes
			curLayer.put(mergeNode, union);
			tree.add(mergeNode);

			// update connected map
			if (connected != null) {
				// 1. merge values of c1 and c2 and put union
				Set<TreeNode> ns = connected.remove(c1);
				ns.addAll(connected.remove(c2));
				connected.put(mergeNode, ns);

				// 2. replace all values c1,c2 by merged node
				for (Set<TreeNode> s : connected.values()) {
					if (s.contains(c1) || s.contains(c2)) {
						s.remove(c1);
						s.remove(c2);
						s.add(mergeNode);
					}
				}
			}
			ssCache.remove(c1);
			ssCache.remove(c2);
			if (hct == HierarchicalClusteringType.ward && !SEC_PHASE) {
				ssCache.put(mergeNode, unionCache.get(c1).get(c2));
			}
		}
		return tree;
	}

	public static double getSumOfSquares(List<Double> residuals) {
		double s = 0;
		for (double d : residuals)
			s += Math.pow(d, 2);
		return s;
	}

	static double getR2(double ssRes, List<double[]> samples, int ta) {
		SummaryStatistics ss = new SummaryStatistics();
		for (double[] d : samples)
			ss.addValue(d[ta]);

		double mean = 0;
		for (double[] d : samples)
			mean += d[ta];
		mean /= samples.size();

		double ssTot = 0;
		for (double[] d : samples)
			ssTot += Math.pow(d[ta] - mean, 2);

		return 1.0 - ssRes / ssTot;

	}

	// H0: equations are equivalent
	private static double[] chowTest(double sc, double s1, double s2, int n1, int n2, int k) {
		double t = ((sc - (s1 + s2)) / k) / ((s1 + s2) / (n1 + n2 - 2 * k));
		FDistribution fd = new FDistribution(k, n1 + n2 - 2 * k);
		return new double[] { t, 1 - fd.cumulativeProbability(t) // p-Value < 0.5 H0(equivalence) rejected, A and B not equal
		};
	}

	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}

	public static double[][] getX(List<double[]> samples, int[] fa, boolean addIntercept) {
		double[][] x = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++) {
			double[] stripped = getStripped(samples.get(i), fa);
			if (addIntercept) {
				stripped = Arrays.copyOf(stripped, stripped.length + 1);
				stripped[stripped.length - 1] = 1;
			}
			x[i] = stripped;
		}
		return x;
	}

	public static double[][] getX(List<Set<double[]>> cluster, List<double[]> samples, int[] fa, boolean addIntercept) {
		double[][] x = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);

			double[] stripped = getStripped(d, fa);
			if (addIntercept) {
				stripped = Arrays.copyOf(stripped, stripped.length + 1);
				stripped[stripped.length - 1] = 1;
			}

			x[i] = new double[stripped.length * cluster.size()];
			for (int j = 0; j < cluster.size(); j++)
				if (cluster.get(j).contains(d)) {
					for (int k = 0; k < stripped.length; k++)
						x[i][stripped.length * j + k] = stripped[k];
					break;
				}
		}
		return x;
	}

	public static double[] getY(List<double[]> samples, int ta) {
		double[] y = new double[samples.size()];
		for (int i = 0; i < samples.size(); i++)
			y[i] = samples.get(i)[ta];
		return y;
	}

	public static class MyOLS extends OLSMultipleLinearRegression {
		@Override
		protected void validateSampleData(double[][] x, double[] y) throws MathIllegalArgumentException {
			if ((x == null) || (y == null)) {
				throw new NullArgumentException();
			}
			if (x.length != y.length) {
				throw new DimensionMismatchException(y.length, x.length);
			}
			if (x.length == 0) { // Must be no y data either
				throw new NoDataException();
			}
			if ((isNoIntercept() ? x[0].length : x[0].length + 1) > x.length) {
				throw new MathIllegalArgumentException(LocalizedFormats.NOT_ENOUGH_DATA_FOR_NUMBER_OF_PREDICTORS, x.length, x[0].length);
			}
		}
	}

	enum StructChangeTestMode {
		Chow, Wald, AChow
	};

	enum Method {
		GWR, StructBreak
	};

	// Wald H0: equations are equivalent
	public static double testStructChange(double[][] x1, double[] y1, double[][] x2, double[] y2, StructChangeTestMode sctm) {
		double T1 = x1.length;
		double T2 = x2.length;
		double T = T1 + T2;
		double k = x1[0].length;

		boolean noIntercept = true;
		OLSMultipleLinearRegression ols1 = new MyOLS();
		ols1.setNoIntercept(noIntercept);
		ols1.newSampleData(y1, x1);

		OLSMultipleLinearRegression ols2 = new MyOLS();
		ols2.setNoIntercept(noIntercept);
		ols2.newSampleData(y2, x2);

		RealMatrix b1 = new Array2DRowRealMatrix(ols1.estimateRegressionParameters());
		RealMatrix b2 = new Array2DRowRealMatrix(ols2.estimateRegressionParameters());
		RealMatrix diff = b1.subtract(b2);

		RealMatrix X1 = new Array2DRowRealMatrix(x1);
		RealMatrix X2 = new Array2DRowRealMatrix(x2);

		RealMatrix m1 = MatrixUtils.inverse(X1.transpose().multiply(X1));
		RealMatrix m2 = MatrixUtils.inverse(X2.transpose().multiply(X2));

		if (sctm == StructChangeTestMode.Chow) {
			double s1 = 0;
			for (double d : ols1.estimateResiduals())
				s1 += d * d;

			double s2 = 0;
			for (double d : ols2.estimateResiduals())
				s2 += d * d;

			double c = diff.transpose().multiply(MatrixUtils.inverse(m1.add(m2))).multiply(diff).getEntry(0, 0) * (T - 2 * k) / (k * (s1 + s2)); // basic chow
			return c;
		} else {
			double s1 = ols1.estimateErrorVariance();
			double s2 = ols2.estimateErrorVariance();
			double w = diff.transpose().multiply(MatrixUtils.inverse(m1.scalarMultiply(s1).add(m2.scalarMultiply(s2)))).multiply(diff).getEntry(0, 0);
			if (sctm == StructChangeTestMode.Wald)
				return w;
			else if (sctm == StructChangeTestMode.AChow)
				return w / k;
		}
		return -1;
	}

	public static List<Double> getResidualsLM(double[][] xTrain, double[] yTrain, double[][] xVal, double[] yVal) {
		OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
		ols.setNoIntercept(true);
		ols.newSampleData(yTrain, xTrain);
		double[] beta = ols.estimateRegressionParameters();

		List<Double> residuals = new ArrayList<>();
		for (int i = 0; i < xVal.length; i++) {
			double[] xi = xVal[i];

			double p = 0;
			for (int j = 0; j < beta.length; j++)
				p += beta[j] * xi[j];

			residuals.add(yVal[i] - p);
		}

		return residuals;
	}

	public static List<Double> getResidualsLM(List<Set<double[]>> cluster, List<double[]> samplesTrain, List<double[]> samplesVal, int[] fa, int ta) {
		List<double[]> s = new ArrayList<double[]>();
		List<Double> residuals = new ArrayList<Double>();

		for (Set<double[]> c : cluster) {
			List<double[]> isTrain = new ArrayList<>(samplesTrain);
			isTrain.retainAll(c);

			List<double[]> isVal = new ArrayList<>(samplesVal);
			isVal.retainAll(c);
			s.addAll(isVal);

			residuals.addAll(getResidualsLM(getX(isTrain, fa, true), getY(isTrain, ta), getX(isVal, fa, true), getY(isVal, ta)));
		}

		// sort residuals
		List<Double> sortedResiduals = new ArrayList<Double>();
		for (double[] d : samplesVal) {
			int idx = s.indexOf(d);
			sortedResiduals.add(residuals.get(idx));
		}
		return sortedResiduals;
	}

	static class ClusterResult {
		ClusterResult(double cost, List<Set<double[]>> cluster, List<Double> residuals, String method) {
			this.cost = cost;
			this.cluster = cluster;
			this.method = method;
			this.residuals = residuals;
		}

		String method;
		double cost;
		List<Set<double[]>> cluster = null;
		List<Double> residuals = null;

		@Override
		public String toString() {
			return method + "," + cluster.size() + "," + cost;
		}
	}
	
	public static int minClusterSize( Collection<Set<double[]>> ct ) {
		int min = Integer.MAX_VALUE;
		for( Set<double[]> s : ct )
			min = Math.min(s.size(), min);
		return min;
	}

	public static double[] getMean(double[][] e) {
		double[] mean = new double[e[0].length];
		for (int r = 0; r < e.length; r++)
			for (int i = 0; i < e[r].length; i++)
				mean[i] += e[r][i] / e.length;
		return mean;
	}

	private static final String KMEANS = "kmeans", LIMIT = "LIMIT", LIMIT2 = "LIMIT2";
	private static String limitString = null;

	public static void main(String[] args) {
		int threads = 12;
		
		//SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/gemeinden/lc2000/merged_dat_gwr.shp"), true);
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/data/gemeinden/lc2000/merged_dat_gwr.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		
		int[] ga = new int[] { 1, 2 };
		int[] fa = new int[] { 0, 4, 5, 6, 7, 9 };
		int[] fa2 = new int[] { 17, 18, 19, 20, 21, 22 };
		int[] fa3 = new int[] { 16, 17, 18, 19, 20, 21, 22 };
		int ta = 10; // lc2000
		//int ta = 12; // lcRate
		
		for( int i = 0; i < fa.length; i++ )
			log.debug("fa "+i+": "+sdf.names.get(fa[i]));
		log.debug("ta: "+ta);
		
		Dist<double[]> gDist = new EuclideanDist(ga);
		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> fDist2 = new EuclideanDist(fa2);
		Dist<double[]> fDist3 = new EuclideanDist(fa3);

		DataUtils.transform(samples, new int[] { 0 }, Transform.sqrt);
		DataUtils.transform(samples, new int[] { 4 }, Transform.log);
		DataUtils.transform(samples, new int[] { ta }, Transform.log);

		List<double[]> samplesOrig = new ArrayList<>();
		for (double[] d : samples)
			samplesOrig.add(Arrays.copyOf(d, d.length));

		// clustering requires standardization
		// DataUtils.transform(samples, new int[] { ta }, Transform.zScore); // should not be necessary
		DataUtils.transform(samples, fa, Transform.zScore);
		DataUtils.transform(samples, fa3, Transform.zScore); // includes fa2
		DataUtils.zScoreGeoColumns(samples, ga, gDist);
		
		Map<double[], Map<double[], Double>> wcm = GraphUtils.toWeightedGraph(GraphUtils.deriveQueenContiguitiyMap(samples, geoms, false), gDist);

		Path file = Paths.get("output/chow.txt");
		try {
			Files.deleteIfExists(file);
			Files.createFile(file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		List<Object[]> params = new ArrayList<>();
		params.add(new Object[] { Method.StructBreak, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, gDist, null, 8, 1 });
		params.add(new Object[] { Method.GWR, HierarchicalClusteringType.ward, null, fDist3, null, -1, 1 }); // with intercept
		
		//params.add(new Object[] { Method.StructBreak, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, gDist, null, 9, 1 });
		//params.add(new Object[] { Method.StructBreak, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, gDist, null, 10, 1 });
		//params.add(new Object[] { Method.GWR, HierarchicalClusteringType.ward, null, fDist2, null, -1, 1 }); // with intercept

		Map<Integer, ClusterResult> re = new HashMap<>();

		for (Object[] param : params) {
			Clustering.r.setSeed(0);

			ClusterResult bestAIC = null;

			double[][] aics;
			double[][] errors;

			int maxRuns = (int) param[6];
			aics = new double[maxRuns][];
			errors = new double[maxRuns][];

			String method = Arrays.toString(param);
			log.debug(method);

			for (int r = 0; r < maxRuns; r++) {
				log.debug(r);

				List<Set<double[]>> init;
				if (param[4] != null && ((String) param[4]).equals(KMEANS)) {
					init = new ArrayList<>(Clustering.kMeans(samples, (int) param[5], (Dist<double[]>) param[3]).values());
				} else {
					init = new ArrayList<>();
					for (double[] d : samples) {
						Set<double[]> s = new HashSet<double[]>();
						s.add(d);
						init.add(s);
					}
				}

				{
					SummaryStatistics ss = new SummaryStatistics();
					for (Set<double[]> s : init)
						ss.addValue(s.size());
					log.debug(ss.getMin() + "," + ss.getMean() + "," + ss.getMax());
				}

				List<TreeNode> curLayer = new ArrayList<>();
				for (Set<double[]> s : init) {
					TreeNode cn = new TreeNode(0, 0);
					cn.setContents(s);
					curLayer.add(cn);
				}

				Map<double[], Set<double[]>> cm = GraphUtils.deriveQueenContiguitiyMap(samples, geoms, false);
				Map<TreeNode, Set<TreeNode>> ncm = new HashMap<>();
				for (TreeNode tnA : curLayer) {
					Set<TreeNode> s = new HashSet<>();
					for (double[] a : tnA.contents)
						for (double[] nb : cm.get(a))
							for (TreeNode tnB : curLayer)
								if (tnB.contents.contains(nb))
									s.add(tnB);
					ncm.put(tnA, s);
				}

				List<TreeNode> tree;
				if ((Method) param[0] == Method.GWR) { // gwr clustering
					tree = LandCon.getHierarchicalClusterTree(samples, cm, (Dist<double[]>) param[3], (HierarchicalClusteringType) param[1], threads);
				} else {
					int threshold = Math.abs((int) param[5]);
					if (param[4] != null && ((String) param[4]).equals(KMEANS))
						threshold = fa.length + 2;

					if (param[4] != null && (((String) param[4]).equals(LIMIT) || ((String) param[4]).equals(LIMIT2)))
						limitString = (String) param[4];
					else
						limitString = null;

					tree = getHierarchicalClusterTree(curLayer, ncm, fa, ta, (HierarchicalClusteringType) param[1], (Dist<double[]>) param[3], threshold, (StructChangeTestMode) param[2], threads);
				}
				log.debug("Tree created...");

				ExecutorService es = Executors.newFixedThreadPool(threads);
				List<Future<List<Set<double[]>>>> futures = new ArrayList<>();
				for (int i = 2; i < Math.min(curLayer.size(), 280); i += 1) { // error goes incredibly up for 280 >, why? 
					final int nrCluster = i;
					futures.add(es.submit(new Callable<List<Set<double[]>>>() {
						@Override
						public List<Set<double[]>> call() throws Exception {
							return Clustering.cutTree(tree, nrCluster);
						}
					}));
				}
				es.shutdown();

				aics[r] = new double[futures.size()];
				errors[r] = new double[futures.size()];

				try {
					for (Future<List<Set<double[]>>> f : futures) {
						List<Set<double[]>> ct = f.get();					
						int idx = futures.indexOf(f);

						List<Double> residuals;
						int nrParams;
						if ( minClusterSize(ct) <= fa.length + 1 ) {
							double[][] x = getX(ct, samples, fa, true);
							double[] y = getY(samples,ta);
							residuals = getResidualsLM(x, y, x, y);
							nrParams = x[0].length;
						} else {
							residuals = getResidualsLM(ct, samples, samples, fa, ta);
							nrParams = ct.size() * (fa.length + 1); 
						}

						double ss = getSumOfSquares(residuals);
						aics[r][idx] = SupervisedUtils.getAICc(ss / samples.size(), nrParams, samples.size());
						errors[r][idx] = Math.sqrt(ss / samples.size());

						if (bestAIC == null || aics[r][idx] < bestAIC.cost) {
							bestAIC = new ClusterResult( aics[r][idx], ct, residuals, "aic_" + method );
						}
					}
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			int idx = params.indexOf(param);
			re.put(idx, bestAIC);

			try {
				String sAIC = idx + ",\"aic," + method + "\"," + Arrays.toString(getMean(aics)).replaceAll("\\[", "").replaceAll("\\]", "") + "\r\n";
				String sError = idx + ",\"error," + method + "\"," + Arrays.toString(getMean(errors)).replaceAll("\\[", "").replaceAll("}}]", "") + "\r\n";

				Files.write(file, sAIC.getBytes(), StandardOpenOption.APPEND);
				Files.write(file, sError.getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		for (Entry<Integer, ClusterResult> e : re.entrySet()) {
			int idx = e.getKey();
			ClusterResult cr = e.getValue();

			Drawer.geoDrawCluster(cr.cluster, samples, geoms, "output/" + idx + ".png", false);

			List<Double> residuals = cr.residuals;
			Map<double[], Double> values = new HashMap<>();
			for (int i = 0; i < samples.size(); i++)
				values.put(samples.get(i), residuals.get(i));
			double[] m1 = GeoUtils.getMoransIStatistics(wcm, values);
			
			double ss = getSumOfSquares(residuals);
			log.info(cr.method);
			log.info("#cluster: " + cr.cluster.size());
			log.info("sse: " + ss);
			log.info("aicc: " + SupervisedUtils.getAICc(ss / samples.size(), cr.cluster.size() * (fa.length + 1), samples.size()));
			log.info("r2: " + getR2(ss, samples, ta));
			log.info("moran: " + Arrays.toString(m1));
			
			List<double[]> l = new ArrayList<double[]>();
			for (double[] d : samples) {
				double[] ns = Arrays.copyOf(d, d.length + 1);
				for (int i = 0; i < cr.cluster.size(); i++) {
					if (cr.cluster.get(i).contains(d)) {
						ns[ns.length - 1] = i;
						break;
					}
				}
				l.add(ns);
			}
			String[] names = sdf.getNames();
			names = Arrays.copyOf(names, names.length + 1);
			names[names.length - 1] = "cluster";

			DataUtils.writeShape(l, geoms, names, sdf.crs, "output/" + idx + ".shp");

			List<String> dissNames = new ArrayList<>();
			for (int i = 0; i < fa.length; i++)
				dissNames.add( names[fa[i]] );
			dissNames.add(  "Intrcpt" );
			
			for (int i = 0; i < fa.length; i++)
				dissNames.add( "p"+names[fa[i]] );
			dissNames.add(  "pIntrcpt" );
			
			dissNames.add(  "Cluster" );
			dissNames.add(  "SSE" );
			
			for (int i = 0; i < fa.length; i++)
				dissNames.add( "std"+names[fa[i]] );
			dissNames.add(  "StdIntrcpt" );
			
			for (int i = 0; i < fa.length; i++)
				dissNames.add( "pStd"+names[fa[i]] );
			dissNames.add(  "pStdIntrcpt" );
			
			dissNames.add(  "StdMaxBeta" );

			PrecisionModel pm = new PrecisionModel(0.1);
			
			List<double[]> dissSamples = new ArrayList<>();
			List<Geometry> dissGeoms = new ArrayList<>();
			for (Set<double[]> s : cr.cluster) {

				List<double[]> li = new ArrayList<>();
				List<Geometry> ggs = new ArrayList<>();
				for (double[] d : s) {
					int idx2 = samples.indexOf(d);
					Geometry g = geoms.get(idx2);
					g = GeometryPrecisionReducer.reduce(g, pm);					
					ggs.add(g);
					
					double[] o = samplesOrig.get(idx2);
					li.add( Arrays.copyOf(o, o.length)); // not standardized
				}
				Geometry union = UnaryUnionOp.union(ggs);
				
				List<Double> dl = new ArrayList<>();
				if( s.size() > fa.length +1 ) {
					TDistribution td = new TDistribution(fa.length+2); // + intercept + error-term
					OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
					ols.setNoIntercept(true);
					ols.newSampleData(getY(li, ta), getX(li, fa, true));
					double[] beta = ols.estimateRegressionParameters();
					double[] err = ols.estimateRegressionParametersStandardErrors();
					double[] pValue = new double[beta.length];
					for( int i = 0; i < beta.length; i++ )
						pValue[i] = 2*(1-td.cumulativeProbability( Math.abs( beta[i]/err[i] ) ) ); 
									
					for( double d : beta )
						dl.add(d);
					for( double d : pValue )
						dl.add(d);
					
					dl.add( (double)(cr.cluster.indexOf(s) ) );
					dl.add( ols.calculateResidualSumOfSquares() );
					
					DataUtils.transform(li, fa, Transform.zScore);
					DataUtils.transform(li, new int[]{ ta }, Transform.zScore);
					ols.newSampleData(getY(li, ta), getX(li, fa, true));
					double[] stdBeta = ols.estimateRegressionParameters();
					double[] stdErr = ols.estimateRegressionParametersStandardErrors();
					double[] stdPValue = new double[stdBeta.length];
					for( int i = 0; i < stdBeta.length; i++ )
						stdPValue[i] = 2*(1-td.cumulativeProbability( Math.abs( stdBeta[i]/stdErr[i] ) ) ); 
					
					for( double d : stdBeta )
						dl.add( d );
					for( double d : stdPValue )
						dl.add( d );
					
					int maxI = 0;
					for (int i = 0; i < stdBeta.length - 1; i++)
						if( Math.abs(stdBeta[i]) > Math.abs(stdBeta[maxI]) )
							maxI = i;
					dl.add( (double)maxI );
					
				} else {
					for( int i = 0; i < (fa.length+1)*4+1 ; i++ )
						dl.add( Double.NaN );
				}
				
				double[] da = new double[dl.size()];
				for( int i = 0; i < dl.size(); i++ )
					da[i] = dl.get(i);

				dissSamples.add(da);
				dissGeoms.add(union);
			}
			DataUtils.writeShape(dissSamples, dissGeoms, dissNames.toArray(new String[]{}), sdf.crs, "output/" + idx + "_diss.shp");
			Drawer.geoDrawValues(dissGeoms, dissSamples, fa.length + 3, sdf.crs, ColorBrewer.Set3, "output/" + idx + "_maxBeta_diss.png");

			for (int i = 0; i < dissNames.size(); i++) {
				String name = dissNames.get(i);
				ColorBrewer cm = ColorBrewer.Blues;
				if( name.contains("Cluster") || name.contains("MaxBeta") )
					cm = ColorBrewer.Set3;
				Drawer.geoDrawValues(dissGeoms, dissSamples, i, sdf.crs, cm, "output/" + idx + "_" + name + "_diss.png");
			}
			
			double ms = 0;
			for( Geometry g : dissGeoms )
				ms += g.getArea()/Math.pow(g.getLength(), 2);
			log.info("A/C^2: "+(ms/dissGeoms.size()));
			
			ms = 0;
			for( Geometry g : dissGeoms ) {
				double a = g.getArea();
				double r = g.getLength()/(Math.PI*2);
				ms += a/(Math.PI*Math.PI*r);
			}
			log.info("Isoperimetric quotient: "+(ms/dissGeoms.size()));
		}
		
		for( int i : re.keySet() )
			for( int j : re.keySet() ) {
				if( i == j )
					continue;
				log.info("NMI "+i+","+j+": " + ClusterValidation.getNormalizedMutualInformation(re.get(i).cluster, re.get(j).cluster) );
			}
	}
}
