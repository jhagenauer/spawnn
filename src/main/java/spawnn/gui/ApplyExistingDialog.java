package spawnn.gui;

import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;

public class ApplyExistingDialog extends JDialog implements ActionListener {

	private JTextField gridField, mapField;
	private JButton gridSel, mapSel;
	private File gridFile, mapFile;
	
	JButton ok, cancel;
	boolean okPressed = false;


	private static final long serialVersionUID = -6577606485481195079L;

	public ApplyExistingDialog(Frame parent, String string, boolean b) {
		super(parent, string, b);

		setLayout(new MigLayout(""));

		add(new JLabel("Grid:"));
		gridField = new JTextField(20);
		gridSel = new JButton("Select...");
		gridSel.addActionListener(this);
		add(gridField,"");
		add(gridSel,"wrap");
		
		add( new JLabel("Mapping:"));
		mapField = new JTextField(20);
		mapSel = new JButton("Select...");
		mapSel.addActionListener(this);
		add(mapField,"");
		add(mapSel,"wrap");
		
		ok = new JButton("OK");
		ok.addActionListener(this);
		cancel = new JButton("Cancel");
		cancel.addActionListener(this);
				
		add(ok, "split 2, push");
		add(cancel);

		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		pack();
		setAlwaysOnTop(true);
		setLocationRelativeTo(parent);
		setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if( e.getSource() == gridSel ) {
			JFileChooser fc = new JFileChooser();
			fc.setFileFilter(FFilter.somXMLFilter);
			if( fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION ) { 
			      gridFile = fc.getSelectedFile();
			      gridField.setText(""+gridFile);
			}
		} else if( e.getSource() == mapSel ) {
			JFileChooser fc = new JFileChooser();
			fc.setFileFilter(FFilter.csvFilter);
			if( fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION ) { 
			      mapFile = fc.getSelectedFile();
			      mapField.setText(""+mapFile);
			}
		} else if (e.getSource() == ok) {
			okPressed = true;
			dispose();
		} else {
			okPressed = false;
			dispose();
		}
	}

	public boolean isOkPressed() {
		return okPressed;
	}
	
	public File getGridFile() {
		return gridFile;
	}
	
	public File getMapFile() {
		return mapFile;
	}
}
