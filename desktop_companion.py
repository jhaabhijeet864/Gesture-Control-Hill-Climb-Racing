#!/usr/bin/env python3
"""
Game Glide Desktop Companion - GUI for profile management and live tuning
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import yaml
import os
import threading
import time
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
from PIL import Image, ImageTk

from plugins.live_config import LiveConfigManager
from plugins.mapping_profiles import ProfileManager
from plugins.plugin_manager import PluginManager
from capture import VideoCapture
from inference import HandInference

class ProfileEditor:
    """Profile editing interface."""
    
    def __init__(self, parent, profile_manager: ProfileManager, on_save_callback=None):
        self.parent = parent
        self.profile_manager = profile_manager
        self.on_save_callback = on_save_callback
        self.current_profile = None
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create the profile editor interface."""
        # Main frame
        self.frame = ttk.LabelFrame(self.parent, text="Profile Editor", padding=10)
        self.frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Profile selection
        selection_frame = ttk.Frame(self.frame)
        selection_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(selection_frame, text="Profile:").pack(side="left")
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(selection_frame, textvariable=self.profile_var, 
                                         state="readonly", width=30)
        self.profile_combo.pack(side="left", padx=(5, 0))
        self.profile_combo.bind("<<ComboboxSelected>>", self.on_profile_selected)
        
        # Buttons
        button_frame = ttk.Frame(selection_frame)
        button_frame.pack(side="right")
        
        ttk.Button(button_frame, text="New", command=self.new_profile).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Load", command=self.load_profile).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Save", command=self.save_profile).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Delete", command=self.delete_profile).pack(side="left", padx=2)
        
        # Profile details
        details_frame = ttk.LabelFrame(self.frame, text="Profile Details", padding=5)
        details_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(details_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.name_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.name_var, width=40).grid(row=0, column=1, sticky="ew")
        
        ttk.Label(details_frame, text="Description:").grid(row=1, column=0, sticky="nw", padx=(0, 5))
        self.desc_text = tk.Text(details_frame, height=3, width=40)
        self.desc_text.grid(row=1, column=1, sticky="ew")
        
        details_frame.columnconfigure(1, weight=1)
        
        # Gesture mappings
        mappings_frame = ttk.LabelFrame(self.frame, text="Gesture Mappings", padding=5)
        mappings_frame.pack(fill="both", expand=True)
        
        # Mappings tree view
        columns = ("Gesture", "Hand", "Action Type", "Action", "Confidence")
        self.mappings_tree = ttk.Treeview(mappings_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.mappings_tree.heading(col, text=col)
            self.mappings_tree.column(col, width=100)
        
        # Scrollbar for tree
        scrollbar = ttk.Scrollbar(mappings_frame, orient="vertical", command=self.mappings_tree.yview)
        self.mappings_tree.configure(yscrollcommand=scrollbar.set)
        
        self.mappings_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mapping controls
        mapping_controls = ttk.Frame(mappings_frame)
        mapping_controls.pack(side="bottom", fill="x", pady=(5, 0))
        
        ttk.Button(mapping_controls, text="Add Mapping", command=self.add_mapping).pack(side="left", padx=2)
        ttk.Button(mapping_controls, text="Edit Mapping", command=self.edit_mapping).pack(side="left", padx=2)
        ttk.Button(mapping_controls, text="Delete Mapping", command=self.delete_mapping).pack(side="left", padx=2)
        
        self.refresh_profiles()
    
    def refresh_profiles(self):
        """Refresh the profile list."""
        profiles = self.profile_manager.list_profiles()
        self.profile_combo['values'] = profiles
        if profiles:
            self.profile_combo.set(profiles[0])
            self.load_current_profile()
    
    def on_profile_selected(self, event=None):
        """Handle profile selection."""
        self.load_current_profile()
    
    def load_current_profile(self):
        """Load the currently selected profile."""
        profile_name = self.profile_var.get()
        if not profile_name:
            return
        
        try:
            self.current_profile = self.profile_manager.load_profile(profile_name)
            self.update_ui_from_profile()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load profile: {e}")
    
    def update_ui_from_profile(self):
        """Update UI elements from the current profile."""
        if not self.current_profile:
            return
        
        self.name_var.set(self.current_profile.name)
        self.desc_text.delete(1.0, tk.END)
        self.desc_text.insert(1.0, getattr(self.current_profile, 'description', ''))
        
        # Clear and populate mappings tree
        for item in self.mappings_tree.get_children():
            self.mappings_tree.delete(item)
        
        for mapping in self.current_profile.mappings:
            values = (
                mapping.gesture,
                getattr(mapping.conditions, 'hand', 'any'),
                mapping.action_type,
                str(mapping.action_params),
                getattr(mapping.conditions, 'confidence', 0.5)
            )
            self.mappings_tree.insert("", "end", values=values)
    
    def new_profile(self):
        """Create a new profile."""
        name = tk.simpledialog.askstring("New Profile", "Enter profile name:")
        if name:
            self.name_var.set(name)
            self.desc_text.delete(1.0, tk.END)
            for item in self.mappings_tree.get_children():
                self.mappings_tree.delete(item)
            self.current_profile = None
    
    def load_profile(self):
        """Load a profile from file."""
        filename = filedialog.askopenfilename(
            title="Load Profile",
            filetypes=[("JSON files", "*.json"), ("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            try:
                # Add to profile manager and refresh
                self.profile_manager.load_profile_from_file(filename)
                self.refresh_profiles()
                messagebox.showinfo("Success", "Profile loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load profile: {e}")
    
    def save_profile(self):
        """Save the current profile."""
        if not self.name_var.get():
            messagebox.showerror("Error", "Please enter a profile name")
            return
        
        try:
            # Create profile data structure
            profile_data = {
                "name": self.name_var.get(),
                "description": self.desc_text.get(1.0, tk.END).strip(),
                "mappings": []
            }
            
            # Get mappings from tree
            for item in self.mappings_tree.get_children():
                values = self.mappings_tree.item(item)['values']
                mapping = {
                    "gesture": values[0],
                    "action_type": values[2],
                    "action_params": eval(values[3]) if values[3] != 'None' else {},
                    "conditions": {
                        "hand": values[1] if values[1] != 'any' else None,
                        "confidence": float(values[4])
                    }
                }
                profile_data["mappings"].append(mapping)
            
            # Save profile
            filename = f"profiles/{self.name_var.get()}.json"
            os.makedirs("profiles", exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            self.refresh_profiles()
            if self.on_save_callback:
                self.on_save_callback()
            
            messagebox.showinfo("Success", f"Profile saved as {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profile: {e}")
    
    def delete_profile(self):
        """Delete the selected profile."""
        profile_name = self.profile_var.get()
        if not profile_name:
            return
        
        if messagebox.askyesno("Confirm Delete", f"Delete profile '{profile_name}'?"):
            try:
                filename = f"profiles/{profile_name}.json"
                if os.path.exists(filename):
                    os.remove(filename)
                self.refresh_profiles()
                messagebox.showinfo("Success", "Profile deleted successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete profile: {e}")
    
    def add_mapping(self):
        """Add a new gesture mapping."""
        MappingDialog(self.frame, callback=self.on_mapping_added)
    
    def edit_mapping(self):
        """Edit the selected mapping."""
        selection = self.mappings_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a mapping to edit")
            return
        
        values = self.mappings_tree.item(selection[0])['values']
        MappingDialog(self.frame, initial_values=values, callback=self.on_mapping_edited)
    
    def delete_mapping(self):
        """Delete the selected mapping."""
        selection = self.mappings_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a mapping to delete")
            return
        
        if messagebox.askyesno("Confirm Delete", "Delete selected mapping?"):
            self.mappings_tree.delete(selection[0])
    
    def on_mapping_added(self, mapping_data):
        """Handle new mapping added."""
        values = (
            mapping_data['gesture'],
            mapping_data['hand'],
            mapping_data['action_type'],
            str(mapping_data['action_params']),
            mapping_data['confidence']
        )
        self.mappings_tree.insert("", "end", values=values)
    
    def on_mapping_edited(self, mapping_data):
        """Handle mapping edited."""
        selection = self.mappings_tree.selection()
        if selection:
            values = (
                mapping_data['gesture'],
                mapping_data['hand'],
                mapping_data['action_type'],
                str(mapping_data['action_params']),
                mapping_data['confidence']
            )
            self.mappings_tree.item(selection[0], values=values)

class MappingDialog:
    """Dialog for adding/editing gesture mappings."""
    
    def __init__(self, parent, initial_values=None, callback=None):
        self.callback = callback
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Gesture Mapping")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets(initial_values)
    
    def create_widgets(self, initial_values):
        """Create dialog widgets."""
        # Gesture
        ttk.Label(self.dialog, text="Gesture:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.gesture_var = tk.StringVar(value=initial_values[0] if initial_values else "")
        gesture_combo = ttk.Combobox(self.dialog, textvariable=self.gesture_var, 
                                   values=["fist", "open_palm", "point", "pinch", "thumbs_up"])
        gesture_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Hand
        ttk.Label(self.dialog, text="Hand:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.hand_var = tk.StringVar(value=initial_values[1] if initial_values else "any")
        hand_combo = ttk.Combobox(self.dialog, textvariable=self.hand_var, 
                                values=["any", "left", "right"])
        hand_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Action Type
        ttk.Label(self.dialog, text="Action Type:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.action_type_var = tk.StringVar(value=initial_values[2] if initial_values else "key")
        action_type_combo = ttk.Combobox(self.dialog, textvariable=self.action_type_var,
                                       values=["key", "mouse_click", "mouse_move", "scroll"])
        action_type_combo.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        # Action Parameters
        ttk.Label(self.dialog, text="Action Params:").grid(row=3, column=0, sticky="nw", padx=5, pady=5)
        self.action_params_text = tk.Text(self.dialog, height=5, width=30)
        self.action_params_text.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        
        if initial_values and len(initial_values) > 3:
            self.action_params_text.insert(1.0, initial_values[3])
        
        # Confidence
        ttk.Label(self.dialog, text="Confidence:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.confidence_var = tk.DoubleVar(value=float(initial_values[4]) if initial_values and len(initial_values) > 4 else 0.7)
        confidence_scale = ttk.Scale(self.dialog, from_=0.1, to=1.0, variable=self.confidence_var, orient="horizontal")
        confidence_scale.grid(row=4, column=1, sticky="ew", padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side="left", padx=5)
        
        self.dialog.columnconfigure(1, weight=1)
    
    def ok_clicked(self):
        """Handle OK button click."""
        try:
            action_params_text = self.action_params_text.get(1.0, tk.END).strip()
            action_params = eval(action_params_text) if action_params_text else {}
        except:
            action_params = {"raw": action_params_text}
        
        mapping_data = {
            'gesture': self.gesture_var.get(),
            'hand': self.hand_var.get(),
            'action_type': self.action_type_var.get(),
            'action_params': action_params,
            'confidence': self.confidence_var.get()
        }
        
        if self.callback:
            self.callback(mapping_data)
        
        self.dialog.destroy()

class LiveConfigPanel:
    """Live configuration tuning panel."""
    
    def __init__(self, parent, config_manager: LiveConfigManager):
        self.parent = parent
        self.config_manager = config_manager
        self.config_vars = {}
        
        self.create_widgets()
        self.load_config_values()
    
    def create_widgets(self):
        """Create configuration widgets."""
        self.frame = ttk.LabelFrame(self.parent, text="Live Configuration", padding=10)
        self.frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create notebook for different config categories
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Detection settings
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="Detection")
        self.create_detection_controls(detection_frame)
        
        # Gesture settings
        gesture_frame = ttk.Frame(self.notebook)
        self.notebook.add(gesture_frame, text="Gestures")
        self.create_gesture_controls(gesture_frame)
        
        # UI settings
        ui_frame = ttk.Frame(self.notebook)
        self.notebook.add(ui_frame, text="UI")
        self.create_ui_controls(ui_frame)
    
    def create_detection_controls(self, parent):
        """Create detection parameter controls."""
        # Min detection confidence
        row = 0
        ttk.Label(parent, text="Min Detection Confidence:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.config_vars['detection.min_detection_confidence'] = tk.DoubleVar()
        scale = ttk.Scale(parent, from_=0.1, to=1.0, 
                         variable=self.config_vars['detection.min_detection_confidence'],
                         orient="horizontal", command=self.on_config_changed)
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        
        # Min tracking confidence
        row += 1
        ttk.Label(parent, text="Min Tracking Confidence:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.config_vars['detection.min_tracking_confidence'] = tk.DoubleVar()
        scale = ttk.Scale(parent, from_=0.1, to=1.0,
                         variable=self.config_vars['detection.min_tracking_confidence'],
                         orient="horizontal", command=self.on_config_changed)
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        
        # Max hands
        row += 1
        ttk.Label(parent, text="Max Number of Hands:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.config_vars['detection.max_num_hands'] = tk.IntVar()
        scale = ttk.Scale(parent, from_=1, to=4,
                         variable=self.config_vars['detection.max_num_hands'],
                         orient="horizontal", command=self.on_config_changed)
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        
        parent.columnconfigure(1, weight=1)
    
    def create_gesture_controls(self, parent):
        """Create gesture parameter controls."""
        # Fist threshold
        row = 0
        ttk.Label(parent, text="Fist Threshold:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.config_vars['gestures.fist_threshold'] = tk.DoubleVar()
        scale = ttk.Scale(parent, from_=0.1, to=1.0,
                         variable=self.config_vars['gestures.fist_threshold'],
                         orient="horizontal", command=self.on_config_changed)
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        
        # Pinch threshold
        row += 1
        ttk.Label(parent, text="Pinch Threshold:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.config_vars['gestures.pinch_threshold'] = tk.DoubleVar()
        scale = ttk.Scale(parent, from_=0.01, to=0.2,
                         variable=self.config_vars['gestures.pinch_threshold'],
                         orient="horizontal", command=self.on_config_changed)
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        
        # Point threshold
        row += 1
        ttk.Label(parent, text="Point Threshold:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.config_vars['gestures.point_threshold'] = tk.DoubleVar()
        scale = ttk.Scale(parent, from_=0.1, to=1.0,
                         variable=self.config_vars['gestures.point_threshold'],
                         orient="horizontal", command=self.on_config_changed)
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        
        parent.columnconfigure(1, weight=1)
    
    def create_ui_controls(self, parent):
        """Create UI parameter controls."""
        # Show landmarks
        row = 0
        ttk.Label(parent, text="Show Landmarks:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.config_vars['ui.show_landmarks'] = tk.BooleanVar()
        checkbox = ttk.Checkbutton(parent, variable=self.config_vars['ui.show_landmarks'],
                                 command=self.on_config_changed)
        checkbox.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        # Show confidence
        row += 1
        ttk.Label(parent, text="Show Confidence:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.config_vars['ui.show_confidence'] = tk.BooleanVar()
        checkbox = ttk.Checkbutton(parent, variable=self.config_vars['ui.show_confidence'],
                                 command=self.on_config_changed)
        checkbox.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        parent.columnconfigure(1, weight=1)
    
    def load_config_values(self):
        """Load current configuration values."""
        for key, var in self.config_vars.items():
            try:
                value = self.config_manager.get_config(key, var.get())
                var.set(value)
            except:
                pass  # Use default value
    
    def on_config_changed(self, *args):
        """Handle configuration change."""
        # Update configuration in real-time
        for key, var in self.config_vars.items():
            try:
                self.config_manager.set_config(key, var.get())
            except Exception as e:
                print(f"Failed to update config {key}: {e}")

class GameGlideCompanion:
    """Main desktop companion application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Game Glide - Desktop Companion")
        self.root.geometry("800x600")
        
        # Initialize managers
        self.config_manager = LiveConfigManager()
        self.profile_manager = ProfileManager()
        
        # UI state
        self.camera_preview_enabled = False
        self.camera_thread = None
        self.preview_label = None
        
        self.create_widgets()
        self.setup_camera_preview()
    
    def create_widgets(self):
        """Create main application widgets."""
        # Create main notebook
        main_notebook = ttk.Notebook(self.root)
        main_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Profile management tab
        profile_frame = ttk.Frame(main_notebook)
        main_notebook.add(profile_frame, text="Profile Management")
        self.profile_editor = ProfileEditor(profile_frame, self.profile_manager)
        
        # Live configuration tab
        config_frame = ttk.Frame(main_notebook)
        main_notebook.add(config_frame, text="Live Configuration")
        self.config_panel = LiveConfigPanel(config_frame, self.config_manager)
        
        # Camera preview tab
        preview_frame = ttk.Frame(main_notebook)
        main_notebook.add(preview_frame, text="Camera Preview")
        self.create_preview_tab(preview_frame)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief="sunken")
        self.status_bar.pack(side="bottom", fill="x")
    
    def create_preview_tab(self, parent):
        """Create camera preview tab."""
        # Controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        self.preview_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Enable Camera Preview", 
                       variable=self.preview_var,
                       command=self.toggle_camera_preview).pack(side="left")
        
        ttk.Button(control_frame, text="Take Screenshot", 
                  command=self.take_screenshot).pack(side="right")
        
        # Preview area
        preview_container = ttk.LabelFrame(parent, text="Camera Preview", padding=5)
        preview_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.preview_label = ttk.Label(preview_container, text="Camera preview disabled")
        self.preview_label.pack(expand=True)
    
    def setup_camera_preview(self):
        """Set up camera preview system."""
        try:
            self.capture = VideoCapture()
            self.inference = HandInference()
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.capture = None
    
    def toggle_camera_preview(self):
        """Toggle camera preview on/off."""
        if self.preview_var.get() and self.capture:
            self.start_camera_preview()
        else:
            self.stop_camera_preview()
    
    def start_camera_preview(self):
        """Start camera preview thread."""
        if self.camera_thread and self.camera_thread.is_alive():
            return
        
        self.camera_preview_enabled = True
        self.camera_thread = threading.Thread(target=self.camera_preview_loop, daemon=True)
        self.camera_thread.start()
        self.status_bar.config(text="Camera preview enabled")
    
    def stop_camera_preview(self):
        """Stop camera preview."""
        self.camera_preview_enabled = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        self.preview_label.config(image="", text="Camera preview disabled")
        self.status_bar.config(text="Camera preview disabled")
    
    def camera_preview_loop(self):
        """Camera preview loop running in separate thread."""
        while self.camera_preview_enabled:
            try:
                ret, frame = self.capture.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Process with hand inference
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.inference.process(rgb)
                
                # Draw landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.inference.draw(frame, hand_landmarks)
                
                # Resize for preview
                height, width = frame.shape[:2]
                max_width = 400
                if width > max_width:
                    scale = max_width / width
                    new_width = max_width
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert to PhotoImage for Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                # Update preview label
                self.preview_label.config(image=photo, text="")
                self.preview_label.image = photo  # Keep reference
                
                time.sleep(1/30)  # ~30 FPS
                
            except Exception as e:
                print(f"Preview error: {e}")
                break
    
    def take_screenshot(self):
        """Take a screenshot of current camera view."""
        if not self.capture:
            messagebox.showerror("Error", "Camera not available")
            return
        
        try:
            ret, frame = self.capture.read()
            if ret:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                cv2.imwrite(filename, cv2.flip(frame, 1))
                messagebox.showinfo("Success", f"Screenshot saved as {filename}")
            else:
                messagebox.showerror("Error", "Failed to capture frame")
        except Exception as e:
            messagebox.showerror("Error", f"Screenshot failed: {e}")
    
    def run(self):
        """Run the application."""
        try:
            self.root.mainloop()
        finally:
            self.stop_camera_preview()
            if hasattr(self, 'capture') and self.capture:
                self.capture.release()

def main():
    """Run the desktop companion application."""
    app = GameGlideCompanion()
    app.run()

if __name__ == "__main__":
    main()
