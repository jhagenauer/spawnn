package spawnn.som.bmu;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Point;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.SpatialDataFrame;

public class GWBmuGetter<T> extends BmuGetter<T> {
	
	private static Logger log = Logger.getLogger(GWBmuGetter.class);

	private Dist<T> fDist, gDist;
	private GWKernel kernel;
	private Map<double[], Double> bandwidth;

	// TODO BW depends on Neuron or Sample?
	public GWBmuGetter(Dist<T> gDist, Dist<T> fDist, GWKernel k, Map<double[], Double> bandwidth) {
		this.gDist = gDist;
		this.fDist = fDist;
		this.kernel = k;
		this.bandwidth = bandwidth;
	}

	@Override
	public GridPos getBmuPos(T x, Grid<T> grid, Set<GridPos> ign) {
		double dist = Double.POSITIVE_INFINITY;

		GridPos bmu = null;
		for (GridPos p : grid.getPositions()) {

			if (ign != null && ign.contains(p))
				continue;
			T v = grid.getPrototypeAt(p);

			double w = GeoUtils.getKernelValue(kernel, gDist.dist(v, x), bandwidth.get(x) );		
			double fd = (1.0-w) * fDist.dist(v, x);
			if (fd < dist) {
				dist = fd;
				bmu = p;
			}
		}

		if (bmu == null)
			throw new RuntimeException("No bmu found: " + grid);

		return bmu;
	}

	public static void main(String[] args) {
		Random r = new Random();
		int T_MAX = 100000;

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"),
				true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}

		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 21, 32, 54, 63, 49 };

		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);

		List<double[]> samples = sdf.samples;
		DataUtils.transform(samples, fa, Transform.zScore);

		log.debug("GeoSOM");
		for (int k : new int[]{ 0, 1, 2, 3, 100 }) {
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);
			BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(gDist, fDist, k);

			SOM som = new SOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}

			log.debug("k: " + k);
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
			log.debug("sqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples));
			log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
		}

		log.debug("GWSOM");
		GWKernel ke = GWKernel.gaussian;
		boolean adaptive = false;
		for (double bw : new double[] { 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0 }) {
			
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);
			
			Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, bw, adaptive);
			BmuGetter<double[]> bmuGetter = new GWBmuGetter<>(gDist, fDist, ke, bandwidth);

			SOM som = new SOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}

			log.debug("bw: " + bw);
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
			log.debug("sqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples));
			log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
		}
	}
}
