package adaptGeoSom;

import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;

public class AdaptKangasBmuGetter<T> extends BmuGetter<T> {

	BmuGetter<T> a, b;
	Map<GridPos, Integer> mk;

	public AdaptKangasBmuGetter(BmuGetter<T> a, BmuGetter<T> b, Map<GridPos, Integer> mk) {
		this.a = a;
		this.b = b;
		this.mk = mk;
	}

	@Override
	public GridPos getBmuPos(T x, Grid<T> grid, Set<GridPos> ign) {
		GridPos bmuA = a.getBmuPos(x, grid);
		int k = mk.get(bmuA);
		if (k == 0)
			return bmuA;

		Set<GridPos> ignore = new HashSet<GridPos>(ign);
		for (GridPos p : grid.getPositions())
			if (grid.dist(bmuA, p) > k)
				ignore.add(p);

		return b.getBmuPos(x, grid, ignore);
	}

	public void setMk(Map<GridPos, Integer> mk) {
		this.mk = mk;
	}
	
	public Map<GridPos, Integer> getMk() {
		return mk;
	}

	public static void main(String[] args) {
		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();

		int X_DIM = 18, Y_DIM = 12;
		int T_MAX = 100000;

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<>();
		Map<Integer, Set<double[]>> cls = new HashMap<>();

		while (samples.size() < 4000) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			int z;
			int c;
			if (x < 0.25) {
				z = 1;
				c = 0;
			} else if (x > 0.75) {
				z = 1;
				c = 1;
			} else {
				z = 0;
				c = 2;
			}

			double[] d = new double[] { x, y, z + r.nextDouble() * 0.2 - 0.1, c, 0 };
			if (!cls.containsKey(c))
				cls.put(c, new HashSet<double[]>());
			cls.get(c).add(d);

			samples.add(d);
			geoms.add(gf.createPoint(new Coordinate(x, y)));
		}

		Dist<double[]> gDist = new EuclideanDist(new int[] { 0, 1 });
		Dist<double[]> fDist = new EuclideanDist(new int[] { 2 });

		DataUtils.transform(samples, new int[] { 2 }, Transform.zScore);

		Grid2D<double[]> grid = new Grid2DHex<double[]>(X_DIM, Y_DIM);
		int maxDist = grid.getMaxDist();
		SomUtils.initRandom(grid, samples);

		BmuGetter<double[]> a = new DefaultBmuGetter<>(gDist);
		BmuGetter<double[]> b = new DefaultBmuGetter<>(fDist);

		Map<GridPos, Integer> mk = new HashMap<>();
		for (GridPos p : grid.getPositions()) {
			double[] pt = grid.getPrototypeAt(p);
			int k = (int) pt[pt.length - 1];
			mk.put(p, k);
		}

		AdaptKangasBmuGetter<double[]> bmuGetter = new AdaptKangasBmuGetter<>(a, b, mk);
		
		SOM som = new SOM(new GaussKernel(new LinearDecay(maxDist, 1)), new LinearDecay(1, 0), grid, bmuGetter);
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size()));

			GridPos geoBmu = a.getBmuPos(x, grid);
			GridPos finalBmu = bmuGetter.getBmuPos(x, grid);

			// idea: increase/decrease k, depending on distance geoBmu-finalBmu
			int d = grid.dist(geoBmu, finalBmu);
			if (geoBmu != finalBmu)
				x[x.length - 1] = 1;
			else
				x[x.length - 1] = maxDist;
			som.train((double) t / T_MAX, x);
			
			mk = new HashMap<>();
			for (GridPos p : grid.getPositions()) {
				double[] pt = grid.getPrototypeAt(p);
				int k = (int)Math.round(pt[pt.length - 1]);
				mk.put(p, k);
			}
			bmuGetter.setMk(mk);
		}
		
		try {
			SomUtils.printComponentPlane(grid, 4, ColorBrewer.Blues, new FileOutputStream("output/k.png"));
			
			SomUtils.printClassDist(cls.values(), SomUtils.getBmuMapping(samples, grid, bmuGetter), grid, "output/classes.png");
			SomUtils.printDMatrix(grid, fDist, "output/umatrix.png");
		} catch( Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("fqe: "+SomUtils.getQuantizationError(grid, bmuGetter, fDist, samples) );
		System.out.println("gqe: "+SomUtils.getQuantizationError(grid, bmuGetter, gDist, samples) );
	}
}
