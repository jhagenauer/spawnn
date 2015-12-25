package wmng.llm;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.zip.GZIPOutputStream;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DToroid;
import spawnn.som.grid.GridPos;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

public class DoubleGrid2DUtils {
	
	private static Logger log = Logger.getLogger(DoubleGrid2DUtils.class);
	
	public static double lim = 0.5;
	public static double noise = 0.2;
	
	public static Grid2D<double[]> createSpDepGrid(int xDim, int yDim, boolean toroid ) {
		
		Random r = new Random();
		Grid2D<double[]> grid;
		if( toroid )
			grid = new Grid2DToroid<double[]>(xDim, yDim);
		else
			grid = new Grid2D<double[]>(xDim, yDim);
		
			List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
			Collections.shuffle(pos);
			
		Map<GridPos,Double> n = new HashMap<GridPos,Double>();
		for( GridPos p : pos ) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			grid.setPrototypeAt(p, new double[]{ i, j, 0, 0 });
			n.put(p, r.nextDouble() );
		}
		
		// two-time avg?! randomly repeatring this might do the trick!!!!!!!!!!!!
		/*Map<GridPos,Double> n2 = new HashMap<GridPos,Double>();
		for( GridPos p : pos ) {
			DescriptiveStatistics dsN = new DescriptiveStatistics();
			dsN.addValue(n.get(p));
			for(GridPos nb : grid.getNeighbours(p) )
				dsN.addValue( n.get(nb));
			n2.put(p, dsN.getMean() );
		}
		n = n2;*/

		for( GridPos p : pos ) {
			
			DescriptiveStatistics dsN = new DescriptiveStatistics();
			dsN.addValue(n.get(p));
			for(GridPos nb : grid.getNeighbours(p) )
				dsN.addValue( n.get(nb) );
			
			// set x_i
			grid.getPrototypeAt(p)[2] = Math.pow( dsN.getMean(), 2) + noise * r.nextDouble(); 
		}
		
		for( GridPos p : pos ) {
			
			DescriptiveStatistics dsX_NBs = new DescriptiveStatistics();
			for(GridPos nb : grid.getNeighbours(p) ) {
				//dsX_NBs.addValue( grid.getPrototypeAt(nb)[2] );
				if( r.nextDouble() < lim) // works
					dsX_NBs.addValue(grid.getPrototypeAt(nb)[2]); 
				else 
					dsX_NBs.addValue(n.get(nb)); // random or n?, n!
			}
			
			grid.getPrototypeAt(p)[3] = Math.pow(grid.getPrototypeAt(p)[2],2) + Math.pow(dsX_NBs.getMean(),2) + noise * r.nextDouble();
		}
		return grid;
	}
	
	public static List<Geometry> getGeoms(List<GridPos> positions ) {
		GeometryFactory gf = new GeometryFactory();
		List<Geometry> geoms = new ArrayList<Geometry>();
		for( GridPos p : positions ) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			geoms.add( gf.createPolygon(new Coordinate[]{
					new Coordinate(i*50, j*50),
					new Coordinate((i+1)*50, j*50),
					new Coordinate((i+1)*50, (j+1)*50),
					new Coordinate(i*50, (j+1)*50),
					new Coordinate(i*50, j*50),
			}) );
		}
		return geoms;
	}
	
	public static SpatialDataFrame gridToSDF(Grid2D<double[]> grid) {
		GeometryFactory gf = new GeometryFactory();
		List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();
		for( GridPos p : pos ) {
			samples.add( grid.getPrototypeAt(p) );
			
			int x = p.getPos(0);
			int y = p.getPos(1);
			int size = 50;
			geoms.add( gf.createPolygon( new Coordinate[]{
				new Coordinate(x*size,y*size),
				new Coordinate(x*size,(y+1)*size),
				new Coordinate((x+1)*size,(y+1)*size),
				new Coordinate((x+1)*size,y*size),
				new Coordinate(x*size,y*size),
			}));
		}
		SpatialDataFrame sdf = new SpatialDataFrame();
		sdf.geoms = geoms;
		sdf.samples = samples;
		sdf.names = new ArrayList<String>();
		sdf.names.add("xPos");
		sdf.names.add("yPos");
		sdf.names.add("x1");
		sdf.names.add("y");
		return sdf;
	}
		
	public static void main(String[] args) {
		DecimalFormat df = new DecimalFormat("000");
		
		for( int i = 0; i < 1; i++ ){
			log.debug(i);
			Grid2D<double[]> grid = createSpDepGrid(50,50,true);
			try {
				GZIPOutputStream gzos = new GZIPOutputStream(new FileOutputStream("output/grid_"+df.format(i)+".xml.gz"));
				SomUtils.saveGrid(grid,  gzos );
				gzos.finish();
				gzos.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			if( i == 0 ) {
				SpatialDataFrame sdf = gridToSDF(grid);
				DataUtils.writeShape(sdf.samples, sdf.geoms, sdf.getNames(), "output/grid.shp");
				
				Map<double[],Map<double[],Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getNeighborsFromGrid(grid)));
				for( int j = 2; j <= 3; j++ ) {
					
					Map<double[],Double> values = new HashMap<double[],Double>();
					for( double[] d : sdf.samples )
						values.put(d,d[j]);
					double[] moran = GeoUtils.getMoransIStatisticsMonteCarlo(dMap, values, 999);
					log.debug("var "+j+", moran: "+moran[0]+", p-Value: "+moran[4]);
				}
			}
		}
	}
}
