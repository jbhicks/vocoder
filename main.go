package main

import (
	"fmt"
	"log"
	"path/filepath"

	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/storage"
	"fyne.io/fyne/v2/widget"
)

func main() {
	a := app.New()
	w := a.NewWindow("Trump Voice Converter")

	inputPath := ""
	outputPath := ""

	inputLabel := widget.NewLabel("Input: (no file selected)")
	outputLabel := widget.NewLabel("Output: (no file selected)")
	statusLabel := widget.NewLabel("Ready")

	inputBtn := widget.NewButton("Select Input WAV", func() {
		fd := dialog.NewFileOpen(func(reader storage.URIReadCloser, err error) {
			if err != nil {
				dialog.ShowError(err, w)
				return
			}
			if reader == nil {
				return
			}
			inputPath = reader.URI().Path()
			inputLabel.SetText(fmt.Sprintf("Input: %s", filepath.Base(inputPath)))
			reader.Close()
		}, w)
		fd.SetFilter(storage.NewExtensionFileFilter([]string{".wav"}))
		fd.Show()
	})

	outputBtn := widget.NewButton("Select Output WAV", func() {
		fd := dialog.NewFileSave(func(writer storage.URIWriteCloser, err error) {
			if err != nil {
				dialog.ShowError(err, w)
				return
			}
			if writer == nil {
				return
			}
			outputPath = writer.URI().Path()
			outputLabel.SetText(fmt.Sprintf("Output: %s", filepath.Base(outputPath)))
			writer.Close()
		}, w)
		fd.SetFilter(storage.NewExtensionFileFilter([]string{".wav"}))
		fd.Show()
	})

	convertBtn := widget.NewButton("Convert to Trump Voice", func() {
		if inputPath == "" || outputPath == "" {
			dialog.ShowError(fmt.Errorf("please select input and output files"), w)
			return
		}

		statusLabel.SetText("Loading models...")
		convertBtn.Disable()

		go func() {
			converter, err := NewConverter()
			if err != nil {
				statusLabel.SetText(fmt.Sprintf("Error: %v", err))
				convertBtn.Enable()
				return
			}
			defer converter.Close()

			statusLabel.SetText("Converting...")
			err = converter.Convert(inputPath, outputPath)
			if err != nil {
				statusLabel.SetText(fmt.Sprintf("Error: %v", err))
				convertBtn.Enable()
				return
			}

			statusLabel.SetText("Conversion complete!")
			convertBtn.Enable()
			dialog.ShowInformation("Success", "Voice conversion completed successfully!", w)
		}()
	})

	content := container.NewVBox(
		widget.NewLabel("Convert any WAV file to Donald Trump's voice"),
		widget.NewSeparator(),
		inputBtn,
		inputLabel,
		outputBtn,
		outputLabel,
		widget.NewSeparator(),
		convertBtn,
		widget.NewSeparator(),
		statusLabel,
	)

	w.SetContent(content)
	w.Resize(a.Settings().Theme().Size("default"))
	w.ShowAndRun()
}
