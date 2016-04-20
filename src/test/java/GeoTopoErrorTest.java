


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
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
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class GeoTopoErrorTest {

	private static Logger log = Logger.getLogger(GeoTopoErrorTest.class);
	
	public static void main(String[] args) {
		Random r = new Random();
		int T_MAX = 200000;
		int X_DIM = 12;
		int Y_DIM = 8;
		
		//File file = new File("data/redcap/Election/election2004.shp");
		File file = new File("data/cng/test2c.shp");
		SpatialDataFrame sd = DataUtils.readShapedata(file, new int[] {}, true);
		List<double[]> samples = sd.samples;
		List<Geometry> geoms = sd.geoms;
		
		// build dist matrix and add coordinates to samples
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p1 = geoms.get(i).getCentroid();

			// ugly, but easy
			/*d[0] = p1.getX();
			d[1] = p1.getY();
			
			d[2] = p1.getX();
			d[3] = p1.getY();*/
		}

		final int[] fa = new int[]{ 2 };
		final int[] ga = new int[] { 0, 1 };
		final int[] gaOrig = new int[] { 0, 1 };
		
		DataUtils.zScoreColumn(samples, fa[0]);

		/*List<double[]> samples = new ArrayList<double[]>();
		while( samples.size() < 10000 ) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			if( x < 1.0/3 || x > 2.0/3 )
				samples.add( new double[]{x,y,10} );
			else
				samples.add( new double[]{x,y,0} );
		}
						
		int[] fa = new int[]{2};
		int[] ga = new int[]{0,1};
		final int[] gaOrig = ga;*/
		
		
		final Dist<double[]> fDist = new EuclideanDist(fa);
		final Dist<double[]> gDist = new EuclideanDist(ga);
		
		Map<Dist<double[]>,Double> map = new HashMap<Dist<double[]>,Double>();
		double alpha = 1;
		map.put(fDist, alpha);
		map.put(gDist, 1.0 - alpha);
		Dist<double[]> dist = new WeightedDist<double[]>(map);
		
		int vLength = samples.get(0).length;
		Grid2D<double[]> grid = new Grid2DHex<double[]>(X_DIM, Y_DIM );
		SomUtils.initRandom(grid, samples);
				
		int maxDist = grid.getMaxDist();
		log.debug("max radius: "+maxDist);
		
		BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(dist);
		
		SOM som = new SOM( new GaussKernel( new LinearDecay( maxDist, 1 ) ), new LinearDecay( 1,0.0 ), grid, bmuGetter );
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size()));
			som.train( (double)t/T_MAX, x );
		}
		
		log.debug("f-te: "+SomUtils.getTopoError(grid, new DefaultBmuGetter<double[]>(fDist), samples) );
		log.debug("g-te: "+SomUtils.getTopoError(grid, new DefaultBmuGetter<double[]>(gDist), samples) );
		
		try {
			SomUtils.printDMatrix( grid, fDist, new FileOutputStream( "output/fDmatrix.png" ) );
			SomUtils.printDMatrix( grid, gDist, new FileOutputStream( "output/gDmatrix.png" ) );
			
			
			SomUtils.printUMatrix( grid, fDist, new FileOutputStream( "output/fUMatrix.png" ) );
			SomUtils.printUMatrix( grid, gDist, new FileOutputStream( "output/gUMatrix.png" ) );
			
			SomUtils.printGeoGrid(gaOrig, grid, new FileOutputStream("output/grid.png"));
			
			for( int i = 0; i < vLength; i++ )
				SomUtils.printComponentPlane(grid, i, new FileOutputStream("output/component"+i+".png"));
			
			// topo error per neuron
			Map<GridPos,Integer> tErrors = new HashMap<GridPos,Integer>();
			Map<GridPos,Integer> hits = new HashMap<GridPos,Integer>();
			for( double[] x : samples ) {
				GridPos s_1 = bmuGetter.getBmuPos(x, grid);							
				if( !hits.containsKey(s_1) )
					hits.put(s_1, 0 );
				hits.put(s_1, hits.get(s_1) + 1 );
				
				Set<GridPos> ign = new HashSet<GridPos>();
				ign.add(s_1);
				GridPos s_2 = bmuGetter.getBmuPos(x, grid, ign);
				
				if( !tErrors.containsKey(s_1) )
					tErrors.put(s_1, 0 );
				if( !grid.getNeighbours(s_1).contains(s_2) ) 
					tErrors.put(s_1, tErrors.get(s_1) + 1 );			
			}
			
			double[][] m = new double[grid.getSizeOfDim(0)][grid.getSizeOfDim(1)];
			for( GridPos p : hits.keySet() ) {
				int[] pos = p.getPosVector();
				m[pos[0]][pos[1]] = (double)(tErrors.get(p))/hits.get(p);
				//log.debug( m[pos[0]][pos[1]]+","+tErrors.get(p)+","+hits.get(p) );
			}		
			SomUtils.printImage( SomUtils.getHexMatrixImage( m, 5, ColorBrewer.Blues, SomUtils.HEX_NORMAL ), new FileOutputStream("output/terror.png") );
	
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
				
		Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter);
		Map<GridPos,Double> fError = new HashMap<GridPos,Double>();
		Map<GridPos,Double> gError = new HashMap<GridPos,Double>();
		
		for( GridPos p : mapping.keySet() ) {
			fError.put( p, DataUtils.getQuantizationError(grid.getPrototypeAt(p), mapping.get(p), fDist) );
			gError.put( p, DataUtils.getQuantizationError(grid.getPrototypeAt(p), mapping.get(p), gDist) );
		}
				
		List<double[]> l = new ArrayList<double[]>();	
		for( double[] d : samples )
			for( GridPos p : mapping.keySet() ) 
				if( mapping.get(p).contains(d) )
					l.add( new double[]{ fError.get(p), gError.get(p) } );
			
		//DataUtils.writeToShape(l, geoms, new String[]{"fError", "gError"}, sd.crs, "output/error.shp");
		
	}
}
