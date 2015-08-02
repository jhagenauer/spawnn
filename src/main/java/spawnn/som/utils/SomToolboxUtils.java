package spawnn.som.utils;


import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import spawnn.dist.Dist;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.utils.DataUtils;

public class SomToolboxUtils {
	
	// it seem's that the dist is irrelevant for the resulting map
	public static void writeUnitDescriptions( Grid2D<double[]> grid, List<double[]> samples, Map<GridPos,Set<double[]>> bmus, Dist<double[]> dist, OutputStream os ) {
		OutputStreamWriter fsw = new OutputStreamWriter(os);
		try {
			fsw.write("$TYPE som\n");
			if( grid instanceof Grid2DHex )
				fsw.write("$GRID_LAYOUT hexagonal\n");
			else 
				fsw.write("$GRID_LAYOUT rectangular\n");
			fsw.write("$GRID_TOPOLOGY planar\n");
			fsw.write("$FILE_FORMAT_VERSION 1.2\n");
			fsw.write("$XDIM "+grid.getSizeOfDim(0)+"\n");
			fsw.write("$YDIM "+grid.getSizeOfDim(1)+"\n");
						
			for( GridPos pos : grid.getPositions() ) {
								
				fsw.write("$POS_X "+pos.getPosVector()[0]+"\n");
				fsw.write("$POS_Y "+pos.getPosVector()[1]+"\n");
				fsw.write("$UNIT_ID u_("+pos.getPosVector()[0]+"/"+pos.getPosVector()[1]+")\n"); // optional
				
				if( bmus.containsKey(pos) ) {
					fsw.write("$NR_VEC_MAPPED "+bmus.get(pos).size()+"\n");
					fsw.write("$MAPPED_VECS\n");
					for( double[] d : bmus.get(pos) )
						fsw.write(samples.indexOf(d)+":"+Arrays.toString(d)+"\n");
						
					String s = "$MAPPED_VECS_DIST";
					for( double[] d : bmus.get(pos) )
						s += " "+dist.dist(d, grid.getPrototypeAt(pos) );
					s += "\n";
					fsw.write(s);	
				} else {
					fsw.write("$NR_VEC_MAPPED 0\n");
				}
			}
			fsw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
	
	public static void writeWeightVectors( Grid2D<double[]> grid, OutputStream os ) {
		OutputStreamWriter fsw = new OutputStreamWriter(os);
		
		double[] first = grid.getPrototypeAt( grid.getPositions().iterator().next() );
		try {
			fsw.write("$TYPE som\n");
			if( grid instanceof Grid2DHex )
				fsw.write("$GRID_LAYOUT hexagonal\n");
			else 
				fsw.write("$GRID_LAYOUT rectangular\n");
			fsw.write("$GRID_TOPOLOGY planar\n");
			fsw.write("$XDIM "+grid.getSizeOfDim(0)+"\n");
			fsw.write("$YDIM "+grid.getSizeOfDim(1)+"\n");
			fsw.write("$ZDIM 1\n");
			fsw.write("$VEC_DIM "+first.length+"\n" );
			
			List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
			
			Collections.sort(pos, new Comparator<GridPos>() {
				@Override
				public int compare(GridPos o1, GridPos o2) {
					if( o1.getPosVector()[1] < o2.getPosVector()[1] )
						return -1;
					else if( o1.getPosVector()[1] > o2.getPosVector()[1] )
						return 1;
					else if( o1.getPosVector()[0] < o2.getPosVector()[0] )
						return -1;
					else if( o1.getPosVector()[0] > o2.getPosVector()[0] )
						return 1;
					else 
						return 0;
				}
			});
			
			for( GridPos p : pos ) {
				double[] v = grid.getPrototypeAt(p); 
				String s = "";
				for( double d : v )
					s += d+" ";
				s += "SOM_MAP_u_("+p.getPosVector()[0]+"/"+p.getPosVector()[1]+")\n";
				fsw.write(s);
			}
			fsw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}

	public static void writeVectors( List<double[]> samples, OutputStream os ) {
		OutputStreamWriter fsw = new OutputStreamWriter(os);
		try {
			fsw.write("$TYPE vec\n");
			fsw.write("$XDIM "+samples.size()+"\n");
			fsw.write("$YDIM 1\n");
			fsw.write("$VEC_DIM "+samples.get(0).length+"\n");
			for( int i = 0; i < samples.size(); i++ ) {
				String s = "";
				for( double d : samples.get(i) )
					s += d+" ";
				s += i+"\n";
				fsw.write(s);
			}
			fsw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
	
	public static void writeClassInformation( List<double[]> samples, Map<double[],Integer> classes, OutputStream os ) {
		Set<Integer> cs = new HashSet<Integer>(classes.values());
		
		OutputStreamWriter fsw = new OutputStreamWriter(os);
		try {
			fsw.write("$TYPE class_information\n");
			fsw.write("$NUM_CLASSES "+cs.size()+"\n");
			fsw.write("$XDIM 2\n");
			fsw.write("$YDIM "+samples.size()+"\n");
			for( int i = 0; i < samples.size(); i++ ) 
				fsw.write( i+" "+classes.get(samples.get(i))+"\n" );
			fsw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
	
	public static void writeTemplateVector( List<double[]> samples, OutputStream os ) {
		OutputStreamWriter fsw = new OutputStreamWriter(os);
		int length = samples.get(0).length;
		try {
			fsw.write("$TYPE template\n");
			fsw.write("$XDIM 2\n");
			fsw.write("$YDIM "+samples.size()+"\n");
			fsw.write("$VEC_DIM "+length+"\n");
			for( int i = 0; i < length; i++ ) 
				fsw.write( i+" attr"+i+"\n" );
			fsw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
	
	public static void main( String[] args ) {
		try {
			List<double[]> samples = DataUtils.readCSV( "/home/julian/tmp/dat.csv", new int[]{} );
			SomToolboxUtils.writeVectors(samples, new FileOutputStream("/home/julian/tmp/dat.vec") );
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

}
