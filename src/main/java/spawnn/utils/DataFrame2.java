package spawnn.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class DataFrame2 {
	
	private static Logger log = LogManager.getLogger(DataFrame2.class);
	
	public double[][] samples_num;		// numeric samples
	public String[][] samples_str;		// string samples
	
	public String[] names_num; 
	public String[] names_str;
		
	public DataFrame2 subset_columns( String[] sel_num, String[] sel_str ) {		
		// sel num samples
		double[][] n_samples_num = new double[samples_num.length][sel_num.length];
		for( int i = 0; i < names_num.length; i++ )
			for( int j = 0; j < sel_num.length; j++ ) 
				if( names_num[i].equals(sel_num[j] ) ) 
					for( int k = 0; k < samples_num.length; k++ )
						n_samples_num[k][j] = samples_num[k][i];
			
		// sel str samples
		String[][] n_samples_str = new String[samples_str.length][sel_str.length];
		for( int i = 0; i < names_str.length; i++ )
			for( int j = 0; j < sel_str.length; j++ ) 
				if( names_str[i].equals(sel_str[j] ) ) 
					for( int k = 0; k < samples_str.length; k++ )
						n_samples_str[k][j] = samples_str[k][i];
		
		DataFrame2 ndf = new DataFrame2();
		ndf.samples_num = n_samples_num;
		ndf.names_num = sel_num;
		
		ndf.samples_str = n_samples_str;
		ndf.names_str = sel_str;			
			
		return ndf;				
	}
	
	public DataFrame2 subset_rows( int[] idx ) {
		DataFrame2 ndf = new DataFrame2();
		ndf.samples_num = new double[idx.length][];
		ndf.names_num = names_num;
		
		ndf.samples_str = new String[idx.length][];
		ndf.names_str = names_str;	
		
		for( int i = 0; i < idx.length; i++ ) {
			ndf.samples_num[i] = samples_num[idx[i]];
			ndf.samples_str[i] = samples_str[idx[i]];
		}
			
		return ndf;				
	}
	
	public DataFrame2 random_rows( int nr ) {
		List<Integer> l = new ArrayList<>();
		for( int i = 0; i < nr; i++ )
			l.add(i);
		Collections.shuffle(l);
		l = l.subList(0, nr);
		int[] i = new int[l.size()];
		for( int j = 0; j < l.size(); j++ )
			i[j] = l.get(j);		
		return subset_rows(i);				
	}
	
	public int size() {
		return samples_num.length;
	}
	
	public DataFrame2 complete_cases() {
		
		List<Integer> idx = new ArrayList<>();
		for( int l = 0; l < samples_num.length; l++ ) {
			
			boolean complete = true;
			for( int i = 0; i < samples_num[l].length; i++ ) 
				if( Double.isNaN(samples_num[l][i]))
					complete = false;
			for( int i = 0; i < samples_str[l].length; i++ )
				if( samples_str[l][i]==null )
					complete = false;
			if( complete )
				idx.add(l);				
		}
		
		int[] i = new int[idx.size()];
		for( int j = 0; j < idx.size(); j++ )
			i[j] = idx.get(j);
		
		return subset_rows(i);
	}
	
	public DataFrame2 complete_cases(int[] idx_num, int[] idx_str ) {
		
		List<Integer> idx = new ArrayList<>();
		for( int l = 0; l < samples_num.length; l++ ) {
			
			boolean complete = true;
			for( int i : idx_num ) 
				if( Double.isNaN(samples_num[l][i]))
					complete = false;
			for( int i : idx_str )
				if( samples_str[l][i]==null )
					complete = false;
			if( complete )
				idx.add(l);				
		}
		
		int[] i = new int[idx.size()];
		for( int j = 0; j < idx.size(); j++ )
			i[j] = idx.get(j);
		
		return subset_rows(i);
	}
	
	public static int getIndex(String[] names, String name ) {
		for( int i = 0; i < names.length; i++ )
			if( names[i].equals(name))
				return i;
		return -1;
	}
	
	public int getIndex( boolean str, String name ) {
		if( str ) {
			for( int i = 0; i < names_str.length; i++ )
				if( names_str[i].equals(name))
					return i;
		} else {
			for( int i = 0; i < names_num.length; i++ )
				if( names_num[i].equals(name))
					return i;
		}
		log.warn(name+" not found in df");
		return -1;
		
	}
}
