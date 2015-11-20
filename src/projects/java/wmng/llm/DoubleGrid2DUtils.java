package wmng.llm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DToroid;
import spawnn.som.grid.GridPos;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataFrame;
import spawnn.utils.DataFrame.binding;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

public class DoubleGrid2DUtils {
	
	private static Logger log = Logger.getLogger(DoubleGrid2DUtils.class);

	public static Grid2D<double[]> createDoubleGrid(int xDim, int yDim, int mode, boolean toroid ) {

		Random r = new Random();
		String st;
		Grid2D<double[]> grid;
		if( toroid ) {
			grid = new Grid2DToroid<double[]>(xDim, yDim);
			st = "toroid";
		} else {
			grid = new Grid2D<double[]>(xDim, yDim);
			st = "grid";
		}
		
		Map<GridPos,Double> probSelect = new HashMap<GridPos,Double>();
		for (int i = 0; i < xDim; i++) {
			for (int j = 0; j < yDim; j++) {
				GridPos p = new GridPos(i,j);
				double[] d = new double[]{i,j,0,0};
				grid.setPrototypeAt(p, d);
				
				probSelect.put(p,1.0);
			}
		}
		
		// tournament selection
		List<GridPos> pos = new ArrayList<GridPos>(probSelect.keySet() );
		Collections.shuffle(pos);
		while( true ) {
						
			int sumProb = 0;
			for( double d : probSelect.values() )
				sumProb += d;
				
			double rnd = r.nextDouble()*sumProb;
			double from = 0, to;
			
			for( GridPos p : pos ) {
				to = from + probSelect.get(p);
				
				if( from <= rnd && rnd < to ) { 
					grid.getPrototypeAt(p)[2] = r.nextDouble();
					
					// update probabilities of p and all direct neighbors
					Set<GridPos> toUpdate = new HashSet<GridPos>();
					toUpdate.add(p);
					toUpdate.addAll(grid.getNeighbours(p));
					
					for( GridPos tp : toUpdate) {
						double pr = 1 + grid.getPrototypeAt(tp)[2];
						for( GridPos nb : grid.getNeighbours(tp) )
							pr += grid.getPrototypeAt(nb)[2];
						probSelect.put(tp,Math.pow(pr,mode));
					}	
					break;
				}
				from = to;
			}
			
			double sumValue = 0;
			for( double[] d : grid.getPrototypes() )
				sumValue += d[2];
			
			if( sumValue > Math.sqrt(grid.size() ) *10 )
				break;
		}
		
		// add dependent variable
		for( GridPos p : grid.getPositions() ) {
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( GridPos nb : grid.getNeighbours(p) )
				ds.addValue( grid.getPrototypeAt(nb)[2] );
			
			double[] pt = grid.getPrototypeAt(p);
			pt[3] = ds.getMean()*pt[2];
		}
		
		
		return grid;
	}
	
	public static void main(String[] args) {
		Grid2D<double[]> grid = createDoubleGrid(50,50,3,true);
		
		GeometryFactory gf = new GeometryFactory();
		
		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();
		for( GridPos p : grid.getPositions() ) {
			double[] d = grid.getPrototypeAt(p);
			int i = p.getPos(0);
			int j = p.getPos(1);
			samples.add(d);
			geoms.add( gf.createPolygon(new Coordinate[]{
					new Coordinate(i*50, j*50),
					new Coordinate((i+1)*50, j*50),
					new Coordinate((i+1)*50, (j+1)*50),
					new Coordinate(i*50, (j+1)*50),
					new Coordinate(i*50, j*50),
			}) );
		}
		
		SpatialDataFrame sdf = new SpatialDataFrame();
		sdf.samples = samples;
		sdf.geoms = geoms;
		sdf.names = new ArrayList<String>();
		sdf.names.add("lon");
		sdf.names.add("lat");
		sdf.names.add("x1");
		sdf.names.add("y");
		sdf.bindings = new ArrayList<DataFrame.binding>();
		sdf.bindings.add(binding.Double);
		sdf.bindings.add(binding.Double);
		sdf.bindings.add(binding.Double);
		sdf.bindings.add(binding.Double);	
		
		Drawer.geoDrawValues(sdf.geoms, sdf.samples, 2, null, ColorMode.Blues, "output/grid_x1.png");
		Drawer.geoDrawValues(sdf.geoms, sdf.samples, 3, null, ColorMode.Blues, "output/grid_y.png");
		DataUtils.writeShape(sdf.samples, sdf.geoms, sdf.names.toArray(new String[]{}), sdf.crs, "output/grid.shp");
	}
}
