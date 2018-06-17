package spDepSOM;

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
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;

public class SpDepSOM {
	public static void main(String[] args) {
		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();

		int X_DIM = 14, Y_DIM = 8;
		int T_MAX = 100000;

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<>();
		Map<Integer, Set<double[]>> cls = new HashMap<>();

		while (samples.size() < 1000) {
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

			double[] d = new double[] { x, y, z, c };
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

		BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<>(fDist);
		BmuGetter<double[]> bmuGetterG = new DefaultBmuGetter<>(gDist);
		
		SOM som = new SOM(new GaussKernel(new LinearDecay(maxDist, 1)), new LinearDecay(1, 0), grid, bmuGetter);
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size()));
			som.train((double)t/T_MAX, x);
		}
		
		Grid2D<double[]> teGrid = new Grid2DHex<double[]>(X_DIM, Y_DIM);
		for( GridPos p : teGrid.getPositions() )
			teGrid.setPrototypeAt(p, new double[]{0});
		
		for( double[] d : samples ) {
			GridPos a = bmuGetter.getBmuPos(d, grid);
			GridPos b = bmuGetterG.getBmuPos(d, grid);
			teGrid.getPrototypeAt(a)[0] = teGrid.getPrototypeAt(a)[0]+grid.dist(a, b);
		}
		Map<GridPos,Set<double[]>> mapping =  SomUtils.getBmuMapping(samples, grid, bmuGetter,true);
		try {
			SomUtils.printComponentPlane(teGrid, 0, "output/te.png");
			SomUtils.printDMatrix(grid, fDist, "output/dmatrix.png");
			SomUtils.printUMatrix(grid, fDist, "output/umatrix.png");
			SomUtils.printClassDist(cls.values(),mapping,grid,"output/classDist.png");
		} catch( Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("fqe: "+SomUtils.getQuantizationError(grid, bmuGetter, fDist, samples) );
		System.out.println("gqe: "+SomUtils.getQuantizationError(grid, bmuGetter, gDist, samples) );
	}
}
