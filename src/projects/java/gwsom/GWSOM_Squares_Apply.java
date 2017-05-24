package gwsom;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.GWBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWSOM_Squares_Apply {

	private static Logger log = Logger.getLogger(GWSOM_Squares_Apply.class);

	public static void main(String[] args) {
		Random r = new Random();
		int T_MAX = 100000;

		GeometryFactory gf = new GeometryFactory();
		List<double[]> samples = new ArrayList<>();
		List<Geometry> geoms = new ArrayList<>();
		Map<double[], Integer> classes = new HashMap<>();
		while (samples.size() < 1000) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			double z;
			int c;
			double noise = r.nextDouble() * 0.1 - 0.05;
			if (x < 0.33) {
				z = 1;
				c = 0;
			} else if (x > 0.66) {
				z = 1;
				c = 1;
			} else {
				z = 0;
				c = 2;
			}
			double[] d = new double[] { x, y, z + noise };
			classes.put(d, c);
			samples.add(d);
			geoms.add(gf.createPoint(new Coordinate(x, y)));
		}

		Map<Integer, Set<double[]>> cm = new HashMap<>();
		for (Entry<double[], Integer> e : classes.entrySet()) {
			if (!cm.containsKey(e.getValue()))
				cm.put(e.getValue(), new HashSet<double[]>());
			cm.get(e.getValue()).add(e.getKey());
		}

		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 2 };

		DataUtils.transform(samples, fa, Transform.zScore);
		DataUtils.writeCSV("output/squares.csv", samples,new String[]{"x","y","z"});

		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);

		double maxDist = 0;
		for (int i = 0; i < samples.size() - 1; i++)
			for (int j = i + 1; j < samples.size(); j++)
				maxDist = Math.max(maxDist, gDist.dist(samples.get(i), samples.get(j)));

		boolean adaptive = false;

		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, samples.size());
		for (GWKernel ke : new GWKernel[] { GWKernel.gaussian })
			for (double bw = 0; bw <= 2; bw+=0.01 ) {

				double sumValError = 0;
				int nrValSamples = 0;
				for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
					List<double[]> samplesTrain = new ArrayList<double[]>();
					for (int k : cvEntry.getKey())
						samplesTrain.add(samples.get(k));

					List<double[]> samplesVal = new ArrayList<double[]>();
					for (int k : cvEntry.getValue())
						samplesVal.add(samples.get(k));

					Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samplesVal, gDist, bw, adaptive);
					for (double[] uv : samplesVal)
						sumValError += Math.pow(
								uv[fa[0]] - GeoUtils.getGWMean(samplesTrain, uv, gDist, ke, bandwidth.get(uv))[fa[0]],
								2);
					nrValSamples += samplesVal.size();
				}

				Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, bw, adaptive);
				List<double[]> values = new ArrayList<>();
				double totError = 0;
				for (double[] uv : samples) {
					double[] m = GeoUtils.getGWMean(samples, uv, gDist, ke, bandwidth.get(uv));
					values.add(m);
					totError += Math.pow(m[fa[0]] - uv[fa[0]], 2);
				}

				//Drawer.geoDrawValues(geoms, values, fa[0], null, ColorBrewer.Blues, "output/gwmean_" + ke + "_" + bw + ".png");
				// DataUtils.writeCSV("output/gwmean_"+ke+"_"+bw+".csv", values,new String[]{"x","y","z"});

				log.debug( ke + ", " + bw + "," + (sumValError / nrValSamples) + "," + (totError / samples.size()) + "\t");
			}
		
		System.exit(1);

		// boxcar ~0.18 interesting
		log.debug("GWSOM");
		for (final GWKernel ke : new GWKernel[] { GWKernel.boxcar })
			for (double bw : new double[]{ 0.18 }) {
				
				Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
				SomUtils.initRandom(grid, samples);

				Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, bw, adaptive);
				BmuGetter<double[]> bmuGetter = new GWBmuGetter(gDist, fDist, ke, bandwidth);

				SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
				for (int t = 0; t < T_MAX; t++) {
					double[] x = samples.get(r.nextInt(samples.size()));
					som.train((double) t / T_MAX, x);
				}
				
				log.debug(ke+","+ bw);
				log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
				log.debug("sqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples));
				log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
				
				Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter,true);
				SomUtils.printGeoGrid(ga, grid, "output/gwsom_grid_"+ke+"_"+bw+".png");
				SomUtils.printDMatrix(grid, fDist, "output/gwsom_dmat_"+ke+"_"+bw+".png");
				SomUtils.printClassDist(classes, mapping, grid, "output/gwsom_class_"+ke+"_"+bw+".png");

			}

		log.debug("GeoSOM");
		for (int k : new int[]{ 3 } ) {
			
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);
			BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(gDist, fDist, k);

			SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}
			
			log.debug("k: " + k);
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
			log.debug("sqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples));
			log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
			
			Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter,true);
			SomUtils.printGeoGrid(ga, grid, "output/geosom_grid_"+k+".png");			
			SomUtils.printDMatrix(grid, fDist, "output/geosom_dmat_"+k+".png");
			SomUtils.printClassDist(classes, mapping, grid, "output/geosom_class_"+k+".png");
		}

		log.debug("WeightedSOM");
		for (double w : new double[]{ 0.5 } ) {

			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);

			Map<Dist<double[]>, Double> m = new HashMap<>();
			m.put(gDist, 1.0 - w);
			m.put(fDist, w);
			Dist<double[]> wDist = new WeightedDist<>(m);
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<>(wDist);

			SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}
			log.debug("w: " + w);
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
			log.debug("sqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples));
			log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
			
			Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter,true);
			SomUtils.printGeoGrid(ga, grid, "output/weightedsom_grid_"+w+".png");
			SomUtils.printDMatrix(grid, fDist, "output/weightedsom_dmat_"+w+".png");
			SomUtils.printClassDist(classes, mapping, grid, "output/weightedsom_class_"+w+".png");
		}
	}

	public static Collection<Set<double[]>> getCluster(Grid<double[]> grid, Dist<double[]> dist,
			Map<GridPos, Set<double[]>> mapping) {
		Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
		for (GridPos p : grid.getPositions()) {
			double[] v = grid.getPrototypeAt(p);
			Set<double[]> s = new HashSet<double[]>();
			for (GridPos nb : grid.getNeighbours(p))
				s.add(grid.getPrototypeAt(nb));
			cm.put(v, s);
		}

		List<TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, dist,
				Clustering.HierarchicalClusteringType.ward);
		List<Set<double[]>> clusters = Clustering.treeToCluster(Clustering.cutTree(tree, 3));

		List<Set<GridPos>> gpCluster = new ArrayList<>();
		for (Set<double[]> s : clusters) {
			Set<GridPos> ns = new HashSet<>();
			for (double[] d : s)
				ns.add(grid.getPositionOf(d));
			gpCluster.add(ns);
		}

		Map<double[], Set<double[]>> ll = new HashMap<double[], Set<double[]>>();
		for (Set<GridPos> s : gpCluster) { // for each cluster s
			Set<double[]> l = new HashSet<double[]>();
			for (GridPos p : s) // for each cluster element p
				if (mapping.containsKey(p))
					l.addAll(mapping.get(p));
			if (!l.isEmpty())
				ll.put(DataUtils.getMean(l), l);
		}
		return ll.values();
	}
}
