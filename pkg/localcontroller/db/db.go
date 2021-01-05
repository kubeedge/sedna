package db

import (
	"os"
	"path/filepath"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/pkg/localcontroller/common/constants"
)

// Resource defines resource (e.g., dataset, model, jointinferenceservice) table
type Resource struct {
	gorm.Model
	Name       string `gorm:"unique"`
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	MetaData   string `json:"metadata"`
	Spec       string `json:"spec"`
}

// SaveResource saves resource info in db
func SaveResource(resource *Resource) error {
	var err error
	dbClient := getClient()

	r := Resource{}

	queryResult := dbClient.Where("name = ?", resource.Name).First(&r)

	if queryResult.RowsAffected == 0 {
		if err = dbClient.Create(resource).Error; err != nil {
			klog.Errorf("saved resource(name=%s) failed, error: %v", resource.Name, err)
			return err
		}
	} else {
		r.APIVersion = resource.APIVersion
		r.Kind = resource.Kind
		r.MetaData = resource.MetaData
		r.Spec = resource.Spec
		if err := dbClient.Save(&r).Error; err != nil {
			klog.Errorf("Update resource(name=%s) failed, error: %v", resource.Name, err)
			return err
		}
	}

	return nil
}

// DeleteResource deletes resource info in db
func DeleteResource(name string) error {
	var err error
	dbClient := getClient()

	r := Resource{}

	queryResult := dbClient.Where("name = ?", name).First(&r)

	if queryResult.RowsAffected == 0 {
		return nil
	}

	if err = dbClient.Unscoped().Delete(&r).Error; err != nil {
		klog.Errorf("delete resource(name=%s) to db failed, error: %v", name, err)
		return err
	}

	return nil
}

// getClient gets db client
func getClient() *gorm.DB {
	dbURL := constants.DataBaseURL

	if _, err := os.Stat(dbURL); err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(filepath.Dir(dbURL), os.ModePerm); err != nil {
				klog.Errorf("create fold(url=%s) failed, error: %v", filepath.Dir(dbURL), err)
			}
		}
	}

	db, err := gorm.Open(sqlite.Open(dbURL), &gorm.Config{})
	if err != nil {
		klog.Errorf("try to connect the db failed, error: %v", err)
	}

	_ = db.AutoMigrate(&Resource{})

	return db
}
